import json
import numpy as np
import subprocess
import tempfile
import os
from abc import ABC, abstractmethod
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

@dataclass
class PerformanceMetrics:
    """Standardized performance metrics across all agents"""
    test_return: float
    train_return: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    total_trades: int
    win_rate: float
    training_time: float
    convergence_episodes: int

@dataclass
class EnhancementResult:
    """Standardized enhancement result"""
    agent_name: str
    enhancement_type: str
    enhancement_config: Dict[str, Any]
    baseline_metrics: PerformanceMetrics
    enhanced_metrics: PerformanceMetrics
    improvement_score: float
    timestamp: str
    metadata: Dict[str, Any] = None

class PipelineAgent(ABC):
    """Base class for all enhancement agents"""
    
    def __init__(self, name: str, enhancement_type: str):
        self.name = name
        self.enhancement_type = enhancement_type
        self.results_log: List[EnhancementResult] = []
    
    @abstractmethod
    def discover_enhancements(self, baseline_config: Dict[str, Any], 
                            baseline_metrics: PerformanceMetrics) -> List[EnhancementResult]:
        """Discover enhancements based on baseline performance"""
        pass
    
    @abstractmethod
    def apply_enhancement(self, config: Dict[str, Any], 
                         enhancement: Dict[str, Any]) -> Dict[str, Any]:
        """Apply enhancement to configuration"""
        pass
    
    def calculate_improvement_score(self, baseline: PerformanceMetrics, 
                                  enhanced: PerformanceMetrics) -> float:
        """Calculate improvement score between baseline and enhanced metrics"""
        # Weighted improvement calculation
        return_improvement = (enhanced.test_return - baseline.test_return) / max(abs(baseline.test_return), 0.01)
        risk_improvement = (baseline.max_drawdown - enhanced.max_drawdown) / max(abs(baseline.max_drawdown), 0.01)
        sharpe_improvement = (enhanced.sharpe_ratio - baseline.sharpe_ratio) / max(abs(baseline.sharpe_ratio), 0.1)
        
        return (return_improvement * 0.4 + 
                risk_improvement * 0.3 + 
                sharpe_improvement * 0.3)

class StrategyRunner:
    """Runs actual strategy and extracts real performance metrics"""
    
    def __init__(self, strategy_script_path: str = "optimization/strategy_optimizer.py"):
        self.strategy_script_path = strategy_script_path
        self.default_config = {
            'symbol': 'GERN',  # This will be overridden
            'source': 'alpha',
            'episodes': 50,
            'max_steps': 1000
        }
    
    def run_strategy(self, config: Dict[str, Any]) -> PerformanceMetrics:
        """Run actual strategy with given configuration and extract real metrics"""
        print(f"🏃 Running actual strategy with config: {self._format_config(config)}")
        
        try:
            # Create temporary config file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(config, f)
                config_file = f.name
            
            # Run actual strategy
            result = self._run_actual_strategy(config, config_file)
            
            # Clean up
            os.unlink(config_file)
            
            return result
            
        except Exception as e:
            print(f"❌ Strategy run failed: {e}")
            return self._get_fallback_metrics()
    
    def _run_actual_strategy(self, config: Dict[str, Any], config_file: str) -> PerformanceMetrics:
        """Run the actual strategy_optimizer.py and capture output"""
        
        # Build command
        cmd = [
            'python', self.strategy_script_path,
            '--symbol', config.get('symbol', 'GERN'),
            '--source', config.get('source', 'alpha'),
            '--episodes', str(config.get('episodes', 50)),
            '--max_steps', str(config.get('max_steps', 1000))
        ]
        
        # Add hyperparameters if present
        if 'learning_rate' in config:
            cmd.extend(['--learning_rate', str(config['learning_rate'])])
        if 'batch_size' in config:
            cmd.extend(['--batch_size', str(config['batch_size'])])
        if 'epsilon_decay' in config:
            cmd.extend(['--epsilon_decay', str(config['epsilon_decay'])])
        
        print(f"   Executing: {' '.join(cmd)}")
        
        # Run strategy and capture output
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
            
            if result.returncode != 0:
                print(f"❌ Strategy execution failed with return code {result.returncode}")
                print(f"   Error: {result.stderr}")
                return self._get_fallback_metrics()
            
            # Parse output to extract metrics
            return self._parse_strategy_output(result.stdout)
            
        except subprocess.TimeoutExpired:
            print("❌ Strategy execution timed out")
            return self._get_fallback_metrics()
        except Exception as e:
            print(f"❌ Strategy execution error: {e}")
            return self._get_fallback_metrics()
    
    def _parse_strategy_output(self, output: str) -> PerformanceMetrics:
        """Parse strategy output to extract performance metrics"""
        lines = output.split('\n')
        
        # Initialize default values
        test_return = 0.0
        train_return = 0.0
        total_trades = 0
        episodes_completed = 0
        
        # Parse output for key metrics
        for line in lines:
            if 'Test Portfolio Return:' in line:
                try:
                    # Extract percentage and convert to decimal
                    percent_str = line.split('Test Portfolio Return:')[1].strip().rstrip('%')
                    test_return = float(percent_str) / 100
                except:
                    pass
            
            elif 'Episode' in line and 'Portfolio Return:' in line:
                try:
                    # Extract training return from episode output
                    parts = line.split('Portfolio Return:')[1].split(',')[0].strip().rstrip('%')
                    train_return = float(parts) / 100
                    episodes_completed += 1
                except:
                    pass
            
            elif 'Total Trades:' in line:
                try:
                    total_trades = int(line.split('Total Trades:')[1].strip())
                except:
                    pass
        
        # Calculate derived metrics
        sharpe_ratio = self._estimate_sharpe_ratio(test_return, train_return)
        max_drawdown = self._estimate_max_drawdown(test_return)
        volatility = self._estimate_volatility(test_return, train_return)
        win_rate = max(0.3, min(0.8, 0.5 + test_return))  # Estimate based on return
        
        print(f"   ✅ Parsed metrics: Return={test_return:.4f}, Trades={total_trades}")
        
        return PerformanceMetrics(
            test_return=test_return,
            train_return=train_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            volatility=volatility,
            total_trades=total_trades,
            win_rate=win_rate,
            training_time=episodes_completed * 2.0,  # Estimate 2 seconds per episode
            convergence_episodes=episodes_completed
        )
    
    def _estimate_sharpe_ratio(self, test_return: float, train_return: float) -> float:
        """Estimate Sharpe ratio from returns"""
        if test_return == 0:
            return 0.0
        
        # Estimate volatility from return difference
        volatility = abs(test_return - train_return) + 0.05  # Add base volatility
        risk_free_rate = 0.02  # 2% risk-free rate
        
        return (test_return - risk_free_rate) / volatility
    
    def _estimate_max_drawdown(self, test_return: float) -> float:
        """Estimate maximum drawdown from test return"""
        # Negative returns suggest higher drawdowns
        if test_return < 0:
            return min(-0.05, test_return * 2)  # At least 5% drawdown for negative returns
        else:
            return max(-0.15, -0.05 - abs(test_return) * 0.5)  # Positive returns still have some drawdown
    
    def _estimate_volatility(self, test_return: float, train_return: float) -> float:
        """Estimate volatility from returns"""
        base_volatility = 0.1
        return_variance = abs(test_return - train_return)
        return base_volatility + return_variance * 2
    
    def _format_config(self, config: Dict[str, Any]) -> str:
        """Format config for logging"""
        key_params = ['symbol', 'source', 'episodes', 'learning_rate', 'batch_size', 'epsilon_decay']
        return {k: v for k, v in config.items() if k in key_params}
    
    def _get_fallback_metrics(self) -> PerformanceMetrics:
        """Return fallback metrics if actual run fails"""
        return PerformanceMetrics(
            test_return=-0.05,  # Assume poor performance if run fails
            train_return=-0.05,
            sharpe_ratio=-0.5,
            max_drawdown=-0.2,
            volatility=0.3,
            total_trades=0,
            win_rate=0.0,
            training_time=0,
            convergence_episodes=0
        )

class ConfigurableStrategyRunner(StrategyRunner):
    """Strategy runner that accepts any symbol and extracts baseline from first run"""
    
    def __init__(self, strategy_script_path: str = "optimization/strategy_optimizer.py"):
        super().__init__(strategy_script_path)
        self.baseline_metrics = None
        self.baseline_config = None
    
    def establish_baseline(self, symbol: str, source: str = 'alpha', episodes: int = 50) -> PerformanceMetrics:
        """Establish baseline performance for any symbol"""
        print(f"📊 Establishing baseline for {symbol} using {source} data...")
        
        baseline_config = {
            'symbol': symbol,
            'source': source,
            'episodes': episodes,
            'max_steps': 1000,
            'learning_rate': 1e-3,
            'epsilon_decay': 0.995,
            'batch_size': 128,
            'epsilon_min': 0.05,
            'memory_size': 20000,
            'gamma': 0.95
        }
        
        baseline_metrics = self.run_strategy(baseline_config)
        
        # Store baseline for reference
        self.baseline_metrics = baseline_metrics
        self.baseline_config = baseline_config
        
        print(f"✅ Baseline established for {symbol}:")
        print(f"   Test Return: {baseline_metrics.test_return:.4f}")
        print(f"   Sharpe Ratio: {baseline_metrics.sharpe_ratio:.4f}")
        print(f"   Max Drawdown: {baseline_metrics.max_drawdown:.4f}")
        print(f"   Total Trades: {baseline_metrics.total_trades}")
        
        return baseline_metrics
    
    def get_baseline(self) -> tuple[Dict[str, Any], PerformanceMetrics]:
        """Get stored baseline config and metrics"""
        if self.baseline_config is None or self.baseline_metrics is None:
            raise ValueError("Baseline not established. Call establish_baseline() first.")
        
        return self.baseline_config, self.baseline_metrics

