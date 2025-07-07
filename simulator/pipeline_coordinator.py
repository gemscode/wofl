from pipeline_base import ConfigurableStrategyRunner, PerformanceMetrics, EnhancementResult
from hyperparameter_agent import HyperparameterAgent
from reward_agent import RewardFunctionAgent
# from indicator_agent import TechnicalIndicatorAgent
# from algorithm_agent import RLAlgorithmAgent
# from risk_agent import RiskManagementAgent
from typing import Dict, Any, List
import json
from datetime import datetime
import os

class PipelineCoordinator:
    def __init__(self, strategy_script_path: str = "optimization/strategy_optimizer.py"):
        self.strategy_runner = ConfigurableStrategyRunner(strategy_script_path)
        self.agents = {
            'hyperparameter': HyperparameterAgent(),
            'reward': RewardFunctionAgent(),
            # Add other agents as you implement them
            # 'indicators': TechnicalIndicatorAgent(),
            # 'algorithms': RLAlgorithmAgent(),
            # 'risk': RiskManagementAgent()
        }
        self.pipeline_results = []
    
    def run_enhancement_pipeline(self, symbol: str, source: str = 'alpha', 
                                episodes: int = 50) -> Dict[str, Any]:
        """Run complete enhancement pipeline for any symbol with real performance metrics"""
        print("🚀 Starting Enhancement Discovery Pipeline")
        print(f"   Symbol: {symbol}")
        print(f"   Source: {source}")
        print(f"   Episodes: {episodes}")
        print("=" * 80)
        
        # Step 1: Establish baseline performance for the specific symbol
        print("📊 STEP 1: Establishing baseline performance...")
        baseline_metrics = self.strategy_runner.establish_baseline(symbol, source, episodes)
        baseline_config, _ = self.strategy_runner.get_baseline()
        
        # Step 2: Run enhancement agents sequentially
        current_config = baseline_config.copy()
        current_metrics = baseline_metrics
        all_enhancements = {}
        improvement_chain = []
        
        for agent_name, agent in self.agents.items():
            print(f"\n🤖 STEP 2.{len(all_enhancements)+1}: Running {agent.name}...")
            print(f"   Current performance: {current_metrics.test_return:.4f}")
            
            try:
                # Agent discovers enhancements based on current performance
                enhancements = agent.discover_enhancements(current_config, current_metrics)
                all_enhancements[agent_name] = enhancements
                
                # Apply best enhancement if it improves performance
                if enhancements and enhancements[0].improvement_score > 0:
                    best_enhancement = enhancements[0]
                    print(f"✅ Applying best enhancement from {agent.name}")
                    print(f"   Enhancement: {best_enhancement.enhancement_config}")
                    print(f"   Improvement score: {best_enhancement.improvement_score:.4f}")
                    
                    # Update current config and metrics for next agent
                    new_config = agent.apply_enhancement(current_config, 
                                                       best_enhancement.enhancement_config)
                    new_metrics = best_enhancement.enhanced_metrics
                    
                    # Record the improvement step
                    improvement_step = {
                        'agent': agent_name,
                        'config_change': best_enhancement.enhancement_config,
                        'performance_before': current_metrics.test_return,
                        'performance_after': new_metrics.test_return,
                        'improvement': new_metrics.test_return - current_metrics.test_return
                    }
                    improvement_chain.append(improvement_step)
                    
                    current_config = new_config
                    current_metrics = new_metrics
                    
                    print(f"   Performance improvement: {improvement_step['improvement']:+.4f}")
                else:
                    print(f"⚠️  No improvements found by {agent.name}")
                    improvement_chain.append({
                        'agent': agent_name,
                        'config_change': None,
                        'performance_before': current_metrics.test_return,
                        'performance_after': current_metrics.test_return,
                        'improvement': 0.0
                    })
                
            except Exception as e:
                print(f"❌ {agent.name} failed: {e}")
                all_enhancements[agent_name] = []
                improvement_chain.append({
                    'agent': agent_name,
                    'config_change': None,
                    'performance_before': current_metrics.test_return,
                    'performance_after': current_metrics.test_return,
                    'improvement': 0.0,
                    'error': str(e)
                })
        
        # Step 3: Final validation with actual run
        print(f"\n📈 STEP 3: Final validation...")
        print(f"   Running final strategy with optimized config...")
        final_metrics = self.strategy_runner.run_strategy(current_config)
        
        # Step 4: Compile comprehensive results
        pipeline_result = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'source': source,
            'episodes': episodes,
            'initial_config': baseline_config,
            'final_config': current_config,
            'baseline_metrics': self._metrics_to_dict(baseline_metrics),
            'final_metrics': self._metrics_to_dict(final_metrics),
            'improvement_chain': improvement_chain,
            'total_improvement': self._calculate_total_improvement(baseline_metrics, final_metrics),
            'agent_enhancements': self._serialize_enhancements(all_enhancements),
            'pipeline_summary': self._generate_summary(baseline_metrics, final_metrics, all_enhancements),
            'config_changes': self._track_config_changes(baseline_config, current_config)
        }
        
        self.pipeline_results.append(pipeline_result)
        self._save_pipeline_results(pipeline_result)
        self._print_final_summary(pipeline_result)
        
        return pipeline_result
    
    def _metrics_to_dict(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Convert PerformanceMetrics to dictionary"""
        return {
            'test_return': metrics.test_return,
            'train_return': metrics.train_return,
            'sharpe_ratio': metrics.sharpe_ratio,
            'max_drawdown': metrics.max_drawdown,
            'volatility': metrics.volatility,
            'total_trades': metrics.total_trades,
            'win_rate': metrics.win_rate,
            'training_time': metrics.training_time,
            'convergence_episodes': metrics.convergence_episodes
        }
    
    def _serialize_enhancements(self, enhancements: Dict[str, List[EnhancementResult]]) -> Dict[str, List[Dict]]:
        """Serialize enhancement results for JSON storage"""
        serialized = {}
        for agent_name, agent_enhancements in enhancements.items():
            serialized[agent_name] = []
            for enhancement in agent_enhancements:
                serialized[agent_name].append({
                    'agent_name': enhancement.agent_name,
                    'enhancement_type': enhancement.enhancement_type,
                    'enhancement_config': enhancement.enhancement_config,
                    'baseline_metrics': self._metrics_to_dict(enhancement.baseline_metrics),
                    'enhanced_metrics': self._metrics_to_dict(enhancement.enhanced_metrics),
                    'improvement_score': enhancement.improvement_score,
                    'timestamp': enhancement.timestamp,
                    'metadata': enhancement.metadata
                })
        return serialized
    
    def _calculate_total_improvement(self, baseline: PerformanceMetrics, 
                                   final: PerformanceMetrics) -> Dict[str, float]:
        """Calculate total improvement across all metrics"""
        return_improvement = final.test_return - baseline.test_return
        sharpe_improvement = final.sharpe_ratio - baseline.sharpe_ratio
        drawdown_improvement = final.max_drawdown - baseline.max_drawdown
        volatility_improvement = baseline.volatility - final.volatility
        
        # Calculate overall improvement score
        overall_score = (return_improvement * 0.4 +
                        sharpe_improvement * 0.2 +
                        (baseline.max_drawdown - final.max_drawdown) * 0.2 +
                        volatility_improvement * 0.2)
        
        return {
            'return_improvement': return_improvement,
            'return_improvement_pct': (return_improvement / max(abs(baseline.test_return), 0.01)) * 100,
            'sharpe_improvement': sharpe_improvement,
            'drawdown_improvement': drawdown_improvement,
            'volatility_improvement': volatility_improvement,
            'overall_score': overall_score,
            'performance_multiplier': final.test_return / baseline.test_return if baseline.test_return != 0 else 1.0
        }
    
    def _generate_summary(self, baseline: PerformanceMetrics, final: PerformanceMetrics,
                         enhancements: Dict[str, List]) -> Dict[str, Any]:
        """Generate comprehensive pipeline summary"""
        total_enhancements = sum(len(agent_enhancements) for agent_enhancements in enhancements.values())
        successful_agents = []
        failed_agents = []
        
        for agent, agent_enhancements in enhancements.items():
            if agent_enhancements and agent_enhancements[0].improvement_score > 0:
                successful_agents.append({
                    'agent': agent,
                    'best_improvement': agent_enhancements[0].improvement_score,
                    'enhancement_config': agent_enhancements[0].enhancement_config
                })
            else:
                failed_agents.append(agent)
        
        return {
            'total_enhancements_tested': total_enhancements,
            'successful_agents': len(successful_agents),
            'failed_agents': len(failed_agents),
            'success_rate': len(successful_agents) / len(self.agents) if self.agents else 0,
            'successful_agent_details': successful_agents,
            'failed_agent_list': failed_agents,
            'performance_change': {
                'return': f"{baseline.test_return:.4f} → {final.test_return:.4f}",
                'sharpe': f"{baseline.sharpe_ratio:.4f} → {final.sharpe_ratio:.4f}",
                'drawdown': f"{baseline.max_drawdown:.4f} → {final.max_drawdown:.4f}",
                'volatility': f"{baseline.volatility:.4f} → {final.volatility:.4f}",
                'trades': f"{baseline.total_trades} → {final.total_trades}"
            },
            'best_performing_agent': max(successful_agents, 
                                       key=lambda x: x['best_improvement'])['agent'] if successful_agents else None
        }
    
    def _track_config_changes(self, initial_config: Dict[str, Any], 
                            final_config: Dict[str, Any]) -> Dict[str, Any]:
        """Track all configuration changes made during pipeline"""
        changes = {}
        
        for key in set(list(initial_config.keys()) + list(final_config.keys())):
            initial_value = initial_config.get(key)
            final_value = final_config.get(key)
            
            if initial_value != final_value:
                changes[key] = {
                    'initial': initial_value,
                    'final': final_value,
                    'changed': True
                }
            else:
                changes[key] = {
                    'value': initial_value,
                    'changed': False
                }
        
        return changes
    
    def _save_pipeline_results(self, result: Dict[str, Any]):
        """Save pipeline results to timestamped file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        symbol = result['symbol']
        filename = f"pipeline_results_{symbol}_{timestamp}.json"
        
        # Ensure results directory exists
        os.makedirs('pipeline_results', exist_ok=True)
        filepath = os.path.join('pipeline_results', filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            print(f"💾 Pipeline results saved to {filepath}")
            
            # Also save a latest results file for easy access
            latest_filepath = os.path.join('pipeline_results', f'latest_{symbol}_results.json')
            with open(latest_filepath, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
    
    def _print_final_summary(self, result: Dict[str, Any]):
        """Print comprehensive final pipeline summary"""
        print("\n🏆 PIPELINE ENHANCEMENT COMPLETE")
        print("=" * 80)
        
        symbol = result['symbol']
        baseline = result['baseline_metrics']
        final = result['final_metrics']
        improvement = result['total_improvement']
        summary = result['pipeline_summary']
        
        print(f"📊 SYMBOL: {symbol}")
        print(f"🕒 COMPLETED: {result['timestamp']}")
        
        print(f"\n📈 PERFORMANCE COMPARISON:")
        print(f"   Return:     {baseline['test_return']:.4f} → {final['test_return']:.4f} "
              f"({improvement['return_improvement']:+.4f}, {improvement['return_improvement_pct']:+.1f}%)")
        print(f"   Sharpe:     {baseline['sharpe_ratio']:.4f} → {final['sharpe_ratio']:.4f} "
              f"({improvement['sharpe_improvement']:+.4f})")
        print(f"   Drawdown:   {baseline['max_drawdown']:.4f} → {final['max_drawdown']:.4f} "
              f"({improvement['drawdown_improvement']:+.4f})")
        print(f"   Volatility: {baseline['volatility']:.4f} → {final['volatility']:.4f} "
              f"({improvement['volatility_improvement']:+.4f})")
        print(f"   Trades:     {baseline['total_trades']} → {final['total_trades']}")
        
        print(f"\n🎯 OVERALL IMPROVEMENT SCORE: {improvement['overall_score']:+.4f}")
        print(f"📊 PERFORMANCE MULTIPLIER: {improvement['performance_multiplier']:.2f}x")
        
        print(f"\n🤖 AGENT PERFORMANCE:")
        print(f"   Total enhancements tested: {summary['total_enhancements_tested']}")
        print(f"   Successful agents: {summary['successful_agents']}/{len(self.agents)} "
              f"({summary['success_rate']:.1%})")
        
        if summary['successful_agent_details']:
            print(f"   Best performing agent: {summary['best_performing_agent']}")
            
            print(f"\n✅ SUCCESSFUL ENHANCEMENTS:")
            for agent_detail in summary['successful_agent_details']:
                print(f"   - {agent_detail['agent']}: {agent_detail['best_improvement']:+.4f} "
                      f"(config: {agent_detail['enhancement_config']})")
        
        if summary['failed_agent_list']:
            print(f"\n❌ AGENTS WITHOUT IMPROVEMENTS:")
            for agent in summary['failed_agent_list']:
                print(f"   - {agent}")
        
        print(f"\n🔄 IMPROVEMENT CHAIN:")
        for i, step in enumerate(result['improvement_chain']):
            status = "✅" if step['improvement'] > 0 else "❌" if step['improvement'] < 0 else "➖"
            print(f"   {i+1}. {step['agent']}: {step['performance_before']:.4f} → "
                  f"{step['performance_after']:.4f} ({step['improvement']:+.4f}) {status}")
        
        # Configuration changes summary
        config_changes = result['config_changes']
        changed_configs = {k: v for k, v in config_changes.items() if v['changed']}
        
        if changed_configs:
            print(f"\n⚙️  CONFIGURATION CHANGES:")
            for param, change in changed_configs.items():
                print(f"   - {param}: {change['initial']} → {change['final']}")
        else:
            print(f"\n⚙️  No configuration changes were beneficial")
        
        print(f"\n💾 Results saved to: pipeline_results/{symbol}_*")
    
    def get_implementation_roadmap(self, result: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """Generate implementation roadmap based on pipeline results"""
        roadmap = {
            'immediate': [],     # Easy wins with high impact
            'short_term': [],    # Medium effort, good impact
            'long_term': [],     # High effort, high impact
            'not_recommended': [] # Low or negative impact
        }
        
        summary = result['pipeline_summary']
        
        for agent_detail in summary['successful_agent_details']:
            enhancement = {
                'agent': agent_detail['agent'],
                'improvement_score': agent_detail['best_improvement'],
                'config': agent_detail['enhancement_config'],
                'implementation_effort': self._estimate_implementation_effort(agent_detail['agent'])
            }
            
            # Categorize based on effort and impact
            if enhancement['improvement_score'] > 0.1 and enhancement['implementation_effort'] == 'low':
                roadmap['immediate'].append(enhancement)
            elif enhancement['improvement_score'] > 0.05 and enhancement['implementation_effort'] in ['low', 'medium']:
                roadmap['short_term'].append(enhancement)
            elif enhancement['improvement_score'] > 0.02:
                roadmap['long_term'].append(enhancement)
            else:
                roadmap['not_recommended'].append(enhancement)
        
        return roadmap
    
    def _estimate_implementation_effort(self, agent_name: str) -> str:
        """Estimate implementation effort for different agent types"""
        effort_map = {
            'hyperparameter': 'low',
            'reward': 'medium',
            'indicators': 'medium',
            'algorithms': 'high',
            'risk': 'low'
        }
        return effort_map.get(agent_name, 'medium')

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run enhancement pipeline for any symbol')
    parser.add_argument('--symbol', type=str, required=True, 
                       help='Symbol to optimize (e.g., GERN, AAPL, TSLA)')
    parser.add_argument('--source', type=str, default='alpha', 
                       choices=['alpha', 'yahoo'], help='Data source')
    parser.add_argument('--episodes', type=int, default=50, 
                       help='Episodes for baseline and testing')
    parser.add_argument('--strategy_path', type=str, default='optimization/strategy_optimizer.py',
                       help='Path to strategy optimizer script')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Pipeline for {args.symbol}")
    print(f"   Source: {args.source}")
    print(f"   Episodes: {args.episodes}")
    print(f"   Strategy script: {args.strategy_path}")
    
    coordinator = PipelineCoordinator(args.strategy_path)
    
    # Run the complete pipeline for any symbol
    results = coordinator.run_enhancement_pipeline(args.symbol, args.source, args.episodes)
    
    # Generate implementation roadmap
    roadmap = coordinator.get_implementation_roadmap(results)
    
    print("\n🗺️  IMPLEMENTATION ROADMAP")
    print("=" * 80)
    for category, enhancements in roadmap.items():
        if enhancements:
            print(f"\n{category.upper().replace('_', ' ')} ({len(enhancements)} items):")
            for enhancement in enhancements:
                print(f"   - {enhancement['agent']}: {enhancement['improvement_score']:+.4f} "
                      f"(effort: {enhancement['implementation_effort']})")
    
    return results

if __name__ == "__main__":
    main()

