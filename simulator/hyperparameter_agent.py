from pipeline_base import PipelineAgent, PerformanceMetrics, EnhancementResult, ConfigurableStrategyRunner
from typing import Dict, Any, List
from datetime import datetime
import itertools
import json

class HyperparameterAgent(PipelineAgent):
    def __init__(self):
        super().__init__("HyperparameterAgent", "hyperparameter_tuning")
        self.strategy_runner = ConfigurableStrategyRunner()
        
        # Define comprehensive hyperparameter search space
        self.search_space = {
            'learning_rate': [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2],
            'epsilon_decay': [0.99, 0.995, 0.998, 0.999],
            'batch_size': [32, 64, 128, 256, 512],
            'epsilon_min': [0.01, 0.05, 0.1, 0.2],
            'gamma': [0.9, 0.95, 0.99, 0.995],
            'memory_size': [10000, 20000, 50000],
            'episodes': [30, 50, 75, 100]
        }
        
        # Define parameter importance for smart search
        self.parameter_importance = {
            'learning_rate': 0.9,
            'epsilon_decay': 0.8,
            'batch_size': 0.7,
            'gamma': 0.6,
            'epsilon_min': 0.5,
            'memory_size': 0.3,
            'episodes': 0.4
        }
    
    def discover_enhancements(self, baseline_config: Dict[str, Any], 
                            baseline_metrics: PerformanceMetrics) -> List[EnhancementResult]:
        """Discover hyperparameter enhancements using actual baseline metrics"""
        print(f"🔧 {self.name}: Starting hyperparameter optimization...")
        print(f"   Baseline return: {baseline_metrics.test_return:.4f}")
        print(f"   Baseline Sharpe: {baseline_metrics.sharpe_ratio:.4f}")
        print(f"   Baseline drawdown: {baseline_metrics.max_drawdown:.4f}")
        
        enhancements = []
        
        # Generate smart parameter combinations
        param_combinations = self._generate_smart_combinations(baseline_config, max_combinations=15)
        
        print(f"   Testing {len(param_combinations)} parameter combinations...")
        
        for i, param_changes in enumerate(param_combinations):
            print(f"\n🧪 Testing combination {i+1}/{len(param_combinations)}")
            print(f"   Changes: {param_changes}")
            
            try:
                # Apply enhancement to baseline config
                enhanced_config = self.apply_enhancement(baseline_config, param_changes)
                
                # Run strategy with enhanced config
                enhanced_metrics = self.strategy_runner.run_strategy(enhanced_config)
                
                # Calculate improvement
                improvement_score = self.calculate_improvement_score(baseline_metrics, enhanced_metrics)
                
                enhancement = EnhancementResult(
                    agent_name=self.name,
                    enhancement_type=self.enhancement_type,
                    enhancement_config=param_changes,
                    baseline_metrics=baseline_metrics,
                    enhanced_metrics=enhanced_metrics,
                    improvement_score=improvement_score,
                    timestamp=datetime.now().isoformat(),
                    metadata={
                        'combination_index': i,
                        'parameter_count': len(param_changes),
                        'expected_impact': self._estimate_parameter_impact(param_changes)
                    }
                )
                
                enhancements.append(enhancement)
                self.results_log.append(enhancement)
                
                print(f"   Result: Return {enhanced_metrics.test_return:.4f} "
                      f"(Δ{enhanced_metrics.test_return - baseline_metrics.test_return:+.4f}), "
                      f"Improvement score: {improvement_score:.4f}")
                
                # Early stopping if we find a very good improvement
                if improvement_score > 0.5:
                    print(f"   🎯 Excellent improvement found! Continuing search...")
                
            except Exception as e:
                print(f"❌ Error with combination {param_changes}: {e}")
                continue
        
        # Sort by improvement score and return top results
        enhancements.sort(key=lambda x: x.improvement_score, reverse=True)
        
        print(f"\n✅ Hyperparameter optimization complete!")
        print(f"   Best improvement: {enhancements[0].improvement_score:.4f}" if enhancements else "No improvements found")
        
        return enhancements[:5]  # Return top 5 improvements
    
    def apply_enhancement(self, config: Dict[str, Any], 
                         enhancement: Dict[str, Any]) -> Dict[str, Any]:
        """Apply hyperparameter enhancement to configuration"""
        enhanced_config = config.copy()
        enhanced_config.update(enhancement)
        return enhanced_config
    
    def _generate_smart_combinations(self, baseline_config: Dict[str, Any], 
                                   max_combinations: int = 15) -> List[Dict[str, Any]]:
        """Generate smart parameter combinations based on baseline and importance"""
        combinations = []
        
        # 1. Single parameter variations (most important first)
        sorted_params = sorted(self.search_space.items(), 
                             key=lambda x: self.parameter_importance.get(x[0], 0.5), 
                             reverse=True)
        
        for param, values in sorted_params:
            baseline_value = baseline_config.get(param)
            for value in values:
                if value != baseline_value:  # Skip if same as baseline
                    combinations.append({param: value})
                    if len(combinations) >= max_combinations // 2:
                        break
            if len(combinations) >= max_combinations // 2:
                break
        
        # 2. Promising multi-parameter combinations
        promising_combos = self._get_promising_combinations(baseline_config)
        combinations.extend(promising_combos[:max_combinations // 3])
        
        # 3. Adaptive combinations based on baseline performance
        adaptive_combos = self._get_adaptive_combinations(baseline_config)
        combinations.extend(adaptive_combos[:max_combinations // 6])
        
        return combinations[:max_combinations]
    
    def _get_promising_combinations(self, baseline_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get combinations known to work well together"""
        promising_combos = [
            # Stable learning combinations
            {'learning_rate': 5e-4, 'batch_size': 256, 'epsilon_decay': 0.998},
            {'learning_rate': 1e-3, 'batch_size': 128, 'gamma': 0.99},
            
            # Exploration-focused combinations
            {'epsilon_decay': 0.999, 'epsilon_min': 0.01, 'memory_size': 50000},
            {'epsilon_decay': 0.995, 'epsilon_min': 0.05, 'learning_rate': 2e-3},
            
            # Conservative combinations
            {'learning_rate': 5e-4, 'epsilon_decay': 0.998, 'gamma': 0.95},
            {'batch_size': 64, 'learning_rate': 1e-3, 'epsilon_min': 0.1},
            
            # Aggressive combinations
            {'learning_rate': 5e-3, 'batch_size': 512, 'gamma': 0.99},
            {'learning_rate': 2e-3, 'epsilon_decay': 0.99, 'episodes': 75},
            
            # Memory-focused combinations
            {'memory_size': 50000, 'batch_size': 256, 'gamma': 0.995},
            {'memory_size': 20000, 'learning_rate': 1e-3, 'epsilon_decay': 0.998}
        ]
        
        # Filter out combinations that are identical to baseline
        filtered_combos = []
        for combo in promising_combos:
            if any(baseline_config.get(k) != v for k, v in combo.items()):
                filtered_combos.append(combo)
        
        return filtered_combos
    
    def _get_adaptive_combinations(self, baseline_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get adaptive combinations based on baseline configuration"""
        adaptive_combos = []
        
        # If baseline learning rate is high, try lower rates with more episodes
        baseline_lr = baseline_config.get('learning_rate', 1e-3)
        if baseline_lr >= 1e-3:
            adaptive_combos.append({'learning_rate': baseline_lr / 2, 'episodes': 75})
            adaptive_combos.append({'learning_rate': baseline_lr / 5, 'batch_size': 256})
        
        # If baseline has few episodes, try more episodes with adjusted parameters
        baseline_episodes = baseline_config.get('episodes', 50)
        if baseline_episodes <= 50:
            adaptive_combos.append({'episodes': 100, 'epsilon_decay': 0.998})
            adaptive_combos.append({'episodes': 75, 'learning_rate': baseline_lr * 0.8})
        
        # If baseline batch size is small, try larger with adjusted learning rate
        baseline_batch = baseline_config.get('batch_size', 128)
        if baseline_batch <= 128:
            adaptive_combos.append({'batch_size': baseline_batch * 2, 'learning_rate': baseline_lr * 0.7})
        
        return adaptive_combos
    
    def _estimate_parameter_impact(self, param_changes: Dict[str, Any]) -> float:
        """Estimate the expected impact of parameter changes"""
        total_impact = 0.0
        
        for param, value in param_changes.items():
            importance = self.parameter_importance.get(param, 0.5)
            
            # Estimate impact based on parameter type and value
            if param == 'learning_rate':
                if 5e-4 <= value <= 2e-3:
                    impact = importance * 0.8  # Good range
                else:
                    impact = importance * 0.4  # Risky range
            elif param == 'batch_size':
                if 64 <= value <= 256:
                    impact = importance * 0.7
                else:
                    impact = importance * 0.3
            elif param == 'epsilon_decay':
                if 0.995 <= value <= 0.999:
                    impact = importance * 0.6
                else:
                    impact = importance * 0.3
            else:
                impact = importance * 0.5  # Default impact
            
            total_impact += impact
        
        return total_impact / len(param_changes)  # Average impact
    
    def calculate_improvement_score(self, baseline: PerformanceMetrics, 
                                  enhanced: PerformanceMetrics) -> float:
        """Calculate improvement score with hyperparameter-specific weighting"""
        # Return improvement (most important for hyperparameter tuning)
        return_improvement = (enhanced.test_return - baseline.test_return) / max(abs(baseline.test_return), 0.01)
        
        # Sharpe ratio improvement
        sharpe_improvement = (enhanced.sharpe_ratio - baseline.sharpe_ratio) / max(abs(baseline.sharpe_ratio), 0.1)
        
        # Risk improvement (drawdown reduction)
        risk_improvement = (baseline.max_drawdown - enhanced.max_drawdown) / max(abs(baseline.max_drawdown), 0.01)
        
        # Training stability (convergence speed)
        stability_improvement = 0.0
        if enhanced.convergence_episodes > 0 and baseline.convergence_episodes > 0:
            stability_improvement = (baseline.convergence_episodes - enhanced.convergence_episodes) / baseline.convergence_episodes
        
        # Weighted combination (emphasize return and stability for hyperparameters)
        improvement_score = (return_improvement * 0.5 + 
                           sharpe_improvement * 0.2 + 
                           risk_improvement * 0.2 + 
                           stability_improvement * 0.1)
        
        return improvement_score
    
    def get_best_enhancements(self) -> List[EnhancementResult]:
        """Return top performing hyperparameter enhancements"""
        if not self.results_log:
            return []
        
        sorted_results = sorted(self.results_log, 
                              key=lambda x: x.improvement_score, 
                              reverse=True)
        return sorted_results[:3]
    
    def save_results(self, filename: str = None):
        """Save hyperparameter optimization results"""
        if filename is None:
            filename = f'{self.name}_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        results_data = {
            'agent_name': self.name,
            'enhancement_type': self.enhancement_type,
            'timestamp': datetime.now().isoformat(),
            'search_space': self.search_space,
            'parameter_importance': self.parameter_importance,
            'total_combinations_tested': len(self.results_log),
            'best_enhancements': [
                {
                    'enhancement_config': result.enhancement_config,
                    'improvement_score': result.improvement_score,
                    'baseline_return': result.baseline_metrics.test_return,
                    'enhanced_return': result.enhanced_metrics.test_return,
                    'return_improvement': result.enhanced_metrics.test_return - result.baseline_metrics.test_return,
                    'metadata': result.metadata
                }
                for result in self.get_best_enhancements()
            ]
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(results_data, f, indent=2)
            print(f"💾 Hyperparameter results saved to {filename}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
    
    def print_summary(self):
        """Print summary of hyperparameter optimization results"""
        if not self.results_log:
            print("No hyperparameter optimization results available")
            return
        
        best_results = self.get_best_enhancements()
        
        print(f"\n📊 HYPERPARAMETER OPTIMIZATION SUMMARY")
        print("=" * 60)
        print(f"Total combinations tested: {len(self.results_log)}")
        print(f"Successful improvements: {len([r for r in self.results_log if r.improvement_score > 0])}")
        
        if best_results:
            print(f"\n🏆 TOP HYPERPARAMETER IMPROVEMENTS:")
            for i, result in enumerate(best_results, 1):
                print(f"\n{i}. Improvement Score: {result.improvement_score:.4f}")
                print(f"   Configuration: {result.enhancement_config}")
                print(f"   Return: {result.baseline_metrics.test_return:.4f} → {result.enhanced_metrics.test_return:.4f}")
                print(f"   Sharpe: {result.baseline_metrics.sharpe_ratio:.4f} → {result.enhanced_metrics.sharpe_ratio:.4f}")
                if result.metadata:
                    print(f"   Expected Impact: {result.metadata.get('expected_impact', 'N/A'):.3f}")

def main():
    """Test the hyperparameter agent"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Hyperparameter Agent')
    parser.add_argument('--symbol', type=str, default='GERN', help='Symbol to test')
    parser.add_argument('--source', type=str, default='alpha', help='Data source')
    parser.add_argument('--episodes', type=int, default=30, help='Episodes for testing')
    
    args = parser.parse_args()
    
    # Create agent
    agent = HyperparameterAgent()
    
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
    
    print(f"🔧 Testing Hyperparameter Agent for {args.symbol}")
    print(f"Establishing baseline...")
    
    baseline_metrics = agent.strategy_runner.run_strategy(baseline_config)
    
    print(f"Baseline established: {baseline_metrics.test_return:.4f}")
    
    # Run hyperparameter optimization
    enhancements = agent.discover_enhancements(baseline_config, baseline_metrics)
    
    # Print results
    agent.print_summary()
    
    # Save results
    agent.save_results()
    
    return enhancements

if __name__ == "__main__":
    main()

