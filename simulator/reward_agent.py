from pipeline_base import PipelineAgent, PerformanceMetrics, EnhancementResult, ConfigurableStrategyRunner
from typing import Dict, Any, List
from datetime import datetime
import json
import numpy as np

class RewardFunctionAgent(PipelineAgent):
    def __init__(self):
        super().__init__("RewardFunctionAgent", "reward_function")
        self.strategy_runner = ConfigurableStrategyRunner()
        
        # Define comprehensive reward function variations
        self.reward_functions = {
            'sharpe_optimized': {
                'description': 'Optimize for Sharpe ratio with risk-adjusted returns',
                'config': {
                    'reward_type': 'sharpe_ratio',
                    'risk_free_rate': 0.02,
                    'volatility_penalty': 0.1,
                    'return_weight': 0.6,
                    'risk_weight': 0.4
                },
                'expected_impact': 0.8
            },
            'drawdown_protected': {
                'description': 'Heavy penalties for large drawdowns',
                'config': {
                    'reward_type': 'drawdown_penalty',
                    'max_drawdown_threshold': 0.1,
                    'drawdown_penalty_multiplier': 2.0,
                    'recovery_bonus': 0.05
                },
                'expected_impact': 0.7
            },
            'consistency_focused': {
                'description': 'Reward consistent returns over volatile ones',
                'config': {
                    'reward_type': 'consistency',
                    'volatility_penalty': 0.2,
                    'consistency_bonus': 0.15,
                    'streak_bonus': 0.1
                },
                'expected_impact': 0.6
            },
            'risk_adjusted': {
                'description': 'Comprehensive risk-adjusted reward function',
                'config': {
                    'reward_type': 'risk_adjusted',
                    'risk_free_rate': 0.02,
                    'var_penalty': 0.15,
                    'skewness_bonus': 0.05,
                    'kurtosis_penalty': 0.03
                },
                'expected_impact': 0.9
            },
            'momentum_enhanced': {
                'description': 'Reward momentum following with trend bonuses',
                'config': {
                    'reward_type': 'momentum',
                    'momentum_threshold': 0.02,
                    'trend_following_bonus': 0.1,
                    'contrarian_penalty': 0.05
                },
                'expected_impact': 0.5
            },
            'profit_factor': {
                'description': 'Optimize profit factor (gross profit / gross loss)',
                'config': {
                    'reward_type': 'profit_factor',
                    'min_profit_factor': 1.5,
                    'win_rate_bonus': 0.1,
                    'large_loss_penalty': 0.2
                },
                'expected_impact': 0.7
            },
            'calmar_ratio': {
                'description': 'Optimize Calmar ratio (return / max drawdown)',
                'config': {
                    'reward_type': 'calmar_ratio',
                    'min_return_threshold': 0.05,
                    'drawdown_scaling': 1.5,
                    'recovery_time_penalty': 0.05
                },
                'expected_impact': 0.8
            },
            'sortino_ratio': {
                'description': 'Optimize Sortino ratio (downside deviation focus)',
                'config': {
                    'reward_type': 'sortino_ratio',
                    'target_return': 0.0,
                    'downside_penalty': 1.5,
                    'upside_bonus': 0.5
                },
                'expected_impact': 0.6
            },
            'kelly_criterion': {
                'description': 'Kelly criterion based position sizing rewards',
                'config': {
                    'reward_type': 'kelly_criterion',
                    'win_rate_weight': 0.4,
                    'avg_win_loss_ratio': 0.6,
                    'overbetting_penalty': 0.3
                },
                'expected_impact': 0.5
            },
            'tail_risk_adjusted': {
                'description': 'Penalize tail risk events heavily',
                'config': {
                    'reward_type': 'tail_risk',
                    'var_confidence': 0.95,
                    'cvar_penalty': 2.0,
                    'extreme_loss_threshold': 0.05,
                    'tail_protection_bonus': 0.1
                },
                'expected_impact': 0.7
            }
        }
        
        # Define reward function complexity and implementation difficulty
        self.implementation_difficulty = {
            'sharpe_optimized': 'low',
            'drawdown_protected': 'low',
            'consistency_focused': 'medium',
            'risk_adjusted': 'medium',
            'momentum_enhanced': 'medium',
            'profit_factor': 'low',
            'calmar_ratio': 'low',
            'sortino_ratio': 'medium',
            'kelly_criterion': 'high',
            'tail_risk_adjusted': 'high'
        }
    
    def discover_enhancements(self, baseline_config: Dict[str, Any], 
                            baseline_metrics: PerformanceMetrics) -> List[EnhancementResult]:
        """Discover reward function enhancements using actual baseline metrics"""
        print(f"💰 {self.name}: Testing reward function variations...")
        print(f"   Baseline return: {baseline_metrics.test_return:.4f}")
        print(f"   Baseline Sharpe ratio: {baseline_metrics.sharpe_ratio:.4f}")
        print(f"   Baseline max drawdown: {baseline_metrics.max_drawdown:.4f}")
        print(f"   Baseline volatility: {baseline_metrics.volatility:.4f}")
        
        enhancements = []
        
        # Sort reward functions by expected impact
        sorted_functions = sorted(self.reward_functions.items(), 
                                key=lambda x: x[1]['expected_impact'], 
                                reverse=True)
        
        print(f"   Testing {len(sorted_functions)} reward function variations...")
        
        for i, (func_name, func_info) in enumerate(sorted_functions):
            print(f"\n🧪 Testing reward function {i+1}/{len(sorted_functions)}: {func_name}")
            print(f"   Description: {func_info['description']}")
            print(f"   Expected impact: {func_info['expected_impact']:.1f}")
            print(f"   Implementation difficulty: {self.implementation_difficulty[func_name]}")
            
            try:
                # Apply reward function enhancement
                enhanced_config = self.apply_enhancement(baseline_config, func_info['config'])
                
                # Run strategy with enhanced reward function
                enhanced_metrics = self.strategy_runner.run_strategy(enhanced_config)
                
                # Calculate improvement with reward-specific weighting
                improvement_score = self._calculate_reward_improvement(baseline_metrics, enhanced_metrics, func_name)
                
                enhancement = EnhancementResult(
                    agent_name=self.name,
                    enhancement_type=self.enhancement_type,
                    enhancement_config=func_info['config'],
                    baseline_metrics=baseline_metrics,
                    enhanced_metrics=enhanced_metrics,
                    improvement_score=improvement_score,
                    timestamp=datetime.now().isoformat(),
                    metadata={
                        'function_name': func_name,
                        'description': func_info['description'],
                        'expected_impact': func_info['expected_impact'],
                        'implementation_difficulty': self.implementation_difficulty[func_name],
                        'risk_focus': self._get_risk_focus(func_name),
                        'return_focus': self._get_return_focus(func_name)
                    }
                )
                
                enhancements.append(enhancement)
                self.results_log.append(enhancement)
                
                print(f"   Result: Return {enhanced_metrics.test_return:.4f} "
                      f"(Δ{enhanced_metrics.test_return - baseline_metrics.test_return:+.4f}), "
                      f"Sharpe {enhanced_metrics.sharpe_ratio:.4f} "
                      f"(Δ{enhanced_metrics.sharpe_ratio - baseline_metrics.sharpe_ratio:+.4f})")
                print(f"   Improvement score: {improvement_score:.4f}")
                
                # Early indication of promising results
                if improvement_score > 0.3:
                    print(f"   🎯 Promising reward function! Continuing evaluation...")
                
            except Exception as e:
                print(f"❌ Error with reward function {func_name}: {e}")
                continue
        
        # Sort by improvement score
        enhancements.sort(key=lambda x: x.improvement_score, reverse=True)
        
        print(f"\n✅ Reward function optimization complete!")
        if enhancements:
            best = enhancements[0]
            print(f"   Best function: {best.metadata['function_name']}")
            print(f"   Best improvement: {best.improvement_score:.4f}")
        else:
            print("   No improvements found")
        
        return enhancements
    
    def apply_enhancement(self, config: Dict[str, Any], 
                         enhancement: Dict[str, Any]) -> Dict[str, Any]:
        """Apply reward function enhancement to configuration"""
        enhanced_config = config.copy()
        
        # Add reward function configuration
        enhanced_config.update(enhancement)
        
        # Add reward function flag to indicate enhanced reward is being used
        enhanced_config['use_enhanced_reward'] = True
        
        return enhanced_config
    
    def _calculate_reward_improvement(self, baseline: PerformanceMetrics, 
                                    enhanced: PerformanceMetrics, 
                                    function_name: str) -> float:
        """Calculate improvement score with reward function specific weighting"""
        
        # Base improvements
        return_improvement = (enhanced.test_return - baseline.test_return) / max(abs(baseline.test_return), 0.01)
        sharpe_improvement = (enhanced.sharpe_ratio - baseline.sharpe_ratio) / max(abs(baseline.sharpe_ratio), 0.1)
        risk_improvement = (baseline.max_drawdown - enhanced.max_drawdown) / max(abs(baseline.max_drawdown), 0.01)
        volatility_improvement = (baseline.volatility - enhanced.volatility) / max(baseline.volatility, 0.01)
        
        # Function-specific weighting
        if function_name in ['sharpe_optimized', 'sortino_ratio']:
            # Emphasize Sharpe ratio improvement
            score = (sharpe_improvement * 0.5 + 
                    return_improvement * 0.3 + 
                    risk_improvement * 0.2)
        
        elif function_name in ['drawdown_protected', 'calmar_ratio', 'tail_risk_adjusted']:
            # Emphasize risk reduction
            score = (risk_improvement * 0.5 + 
                    return_improvement * 0.3 + 
                    sharpe_improvement * 0.2)
        
        elif function_name in ['consistency_focused']:
            # Emphasize volatility reduction and steady returns
            score = (volatility_improvement * 0.4 + 
                    return_improvement * 0.3 + 
                    sharpe_improvement * 0.3)
        
        elif function_name in ['momentum_enhanced']:
            # Emphasize return improvement
            score = (return_improvement * 0.6 + 
                    sharpe_improvement * 0.3 + 
                    risk_improvement * 0.1)
        
        else:  # risk_adjusted, profit_factor, kelly_criterion
            # Balanced approach
            score = (return_improvement * 0.4 + 
                    sharpe_improvement * 0.3 + 
                    risk_improvement * 0.2 + 
                    volatility_improvement * 0.1)
        
        return score
    
    def _get_risk_focus(self, function_name: str) -> float:
        """Get risk focus score for the reward function (0-1)"""
        risk_focus_map = {
            'sharpe_optimized': 0.7,
            'drawdown_protected': 0.9,
            'consistency_focused': 0.8,
            'risk_adjusted': 0.8,
            'momentum_enhanced': 0.3,
            'profit_factor': 0.5,
            'calmar_ratio': 0.9,
            'sortino_ratio': 0.7,
            'kelly_criterion': 0.6,
            'tail_risk_adjusted': 1.0
        }
        return risk_focus_map.get(function_name, 0.5)
    
    def _get_return_focus(self, function_name: str) -> float:
        """Get return focus score for the reward function (0-1)"""
        return_focus_map = {
            'sharpe_optimized': 0.6,
            'drawdown_protected': 0.3,
            'consistency_focused': 0.5,
            'risk_adjusted': 0.6,
            'momentum_enhanced': 0.9,
            'profit_factor': 0.8,
            'calmar_ratio': 0.7,
            'sortino_ratio': 0.6,
            'kelly_criterion': 0.7,
            'tail_risk_adjusted': 0.4
        }
        return return_focus_map.get(function_name, 0.5)
    
    def get_best_enhancements(self) -> List[EnhancementResult]:
        """Return top performing reward function enhancements"""
        if not self.results_log:
            return []
        
        sorted_results = sorted(self.results_log, 
                              key=lambda x: x.improvement_score, 
                              reverse=True)
        return sorted_results[:3]
    
    def save_results(self, filename: str = None):
        """Save reward function optimization results"""
        if filename is None:
            filename = f'{self.name}_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        results_data = {
            'agent_name': self.name,
            'enhancement_type': self.enhancement_type,
            'timestamp': datetime.now().isoformat(),
            'available_functions': list(self.reward_functions.keys()),
            'implementation_difficulty': self.implementation_difficulty,
            'total_functions_tested': len(self.results_log),
            'best_enhancements': [
                {
                    'function_name': result.metadata['function_name'],
                    'description': result.metadata['description'],
                    'enhancement_config': result.enhancement_config,
                    'improvement_score': result.improvement_score,
                    'baseline_return': result.baseline_metrics.test_return,
                    'enhanced_return': result.enhanced_metrics.test_return,
                    'baseline_sharpe': result.baseline_metrics.sharpe_ratio,
                    'enhanced_sharpe': result.enhanced_metrics.sharpe_ratio,
                    'baseline_drawdown': result.baseline_metrics.max_drawdown,
                    'enhanced_drawdown': result.enhanced_metrics.max_drawdown,
                    'risk_focus': result.metadata['risk_focus'],
                    'return_focus': result.metadata['return_focus'],
                    'implementation_difficulty': result.metadata['implementation_difficulty']
                }
                for result in self.get_best_enhancements()
            ]
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(results_data, f, indent=2)
            print(f"💾 Reward function results saved to {filename}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
    
    def print_summary(self):
        """Print summary of reward function optimization results"""
        if not self.results_log:
            print("No reward function optimization results available")
            return
        
        best_results = self.get_best_enhancements()
        
        print(f"\n💰 REWARD FUNCTION OPTIMIZATION SUMMARY")
        print("=" * 60)
        print(f"Total functions tested: {len(self.results_log)}")
        print(f"Successful improvements: {len([r for r in self.results_log if r.improvement_score > 0])}")
        
        if best_results:
            print(f"\n🏆 TOP REWARD FUNCTION IMPROVEMENTS:")
            for i, result in enumerate(best_results, 1):
                metadata = result.metadata
                print(f"\n{i}. {metadata['function_name'].upper()}")
                print(f"   Description: {metadata['description']}")
                print(f"   Improvement Score: {result.improvement_score:.4f}")
                print(f"   Implementation: {metadata['implementation_difficulty']} difficulty")
                print(f"   Focus: {metadata['risk_focus']:.1f} risk, {metadata['return_focus']:.1f} return")
                print(f"   Performance Changes:")
                print(f"     Return: {result.baseline_metrics.test_return:.4f} → {result.enhanced_metrics.test_return:.4f}")
                print(f"     Sharpe: {result.baseline_metrics.sharpe_ratio:.4f} → {result.enhanced_metrics.sharpe_ratio:.4f}")
                print(f"     Drawdown: {result.baseline_metrics.max_drawdown:.4f} → {result.enhanced_metrics.max_drawdown:.4f}")
        
        # Implementation recommendations
        print(f"\n📋 IMPLEMENTATION RECOMMENDATIONS:")
        easy_wins = [r for r in best_results if r.metadata['implementation_difficulty'] == 'low' and r.improvement_score > 0]
        medium_wins = [r for r in best_results if r.metadata['implementation_difficulty'] == 'medium' and r.improvement_score > 0]
        hard_wins = [r for r in best_results if r.metadata['implementation_difficulty'] == 'high' and r.improvement_score > 0]
        
        if easy_wins:
            print(f"   Easy wins ({len(easy_wins)}): {', '.join([r.metadata['function_name'] for r in easy_wins])}")
        if medium_wins:
            print(f"   Medium effort ({len(medium_wins)}): {', '.join([r.metadata['function_name'] for r in medium_wins])}")
        if hard_wins:
            print(f"   High effort ({len(hard_wins)}): {', '.join([r.metadata['function_name'] for r in hard_wins])}")

def main():
    """Test the reward function agent"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Reward Function Agent')
    parser.add_argument('--symbol', type=str, default='GERN', help='Symbol to test')
    parser.add_argument('--source', type=str, default='alpha', help='Data source')
    parser.add_argument('--episodes', type=int, default=30, help='Episodes for testing')
    
    args = parser.parse_args()
    
    # Create agent
    agent = RewardFunctionAgent()
    
    # Establish baseline
    baseline_config = {
        'symbol': args.symbol,
        'source': args.source,
        'episodes': args.episodes,
        'max_steps': 1000,
        'learning_rate': 1e-3,
        'epsilon_decay': 0.995,
        'batch_size': 128,
        'epsilon_min': 0.05,
        'memory_size': 20000,
        'gamma': 0.95
    }
    
    print(f"💰 Testing Reward Function Agent for {args.symbol}")
    print(f"Establishing baseline...")
    
    baseline_metrics = agent.strategy_runner.run_strategy(baseline_config)
    
    print(f"Baseline established: Return {baseline_metrics.test_return:.4f}, "
          f"Sharpe {baseline_metrics.sharpe_ratio:.4f}")
    
    # Run reward function optimization
    enhancements = agent.discover_enhancements(baseline_config, baseline_metrics)
    
    # Print results
    agent.print_summary()
    
    # Save results
    agent.save_results()
    
    return enhancements

if __name__ == "__main__":
    main()

