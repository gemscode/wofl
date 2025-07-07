from ensemble_trading_system import create_ensemble_system, EnsembleDecisionMaker
from pipeline_coordinator import PipelineCoordinator
import torch
import pickle

class LiveTradingSystem:
    """Integration of ensemble system with your pipeline"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.ensemble = None
        self.pipeline_coordinator = PipelineCoordinator()
        
    def initialize_ensemble(self, pipeline_model_path: str = None):
        """Initialize ensemble with pipeline-trained model"""
        
        # Load pipeline model if available
        pipeline_model = None
        if pipeline_model_path:
            try:
                pipeline_model = torch.load(pipeline_model_path)
                print(f"✅ Loaded pipeline model from {pipeline_model_path}")
            except Exception as e:
                print(f"❌ Failed to load pipeline model: {e}")
        
        # Create ensemble system
        self.ensemble = create_ensemble_system(pipeline_model)
        print("✅ Ensemble trading system initialized")
        
        return self.ensemble
    
    def retrain_pipeline_agent(self, episodes: int = 50):
        """Retrain the pipeline agent with recent data"""
        print(f"🔄 Retraining pipeline agent for {self.symbol}...")
        
        # Run pipeline optimization
        results = self.pipeline_coordinator.run_enhancement_pipeline(
            symbol=self.symbol,
            episodes=episodes
        )
        
        # Extract best model (this would need to be implemented in your pipeline)
        # For now, we'll simulate this
        print("✅ Pipeline agent retrained")
        
        return results
    
    def make_live_trading_decision(self, market_data: dict) -> dict:
        """Make live trading decision using ensemble"""
        
        if not self.ensemble:
            raise ValueError("Ensemble not initialized. Call initialize_ensemble() first.")
        
        # Make decision
        action, confidence, details = self.ensemble.make_trading_decision(market_data)
        
        # Format for live trading
        decision = {
            'symbol': self.symbol,
            'action': ['HOLD', 'BUY', 'SELL'][action],
            'confidence': confidence,
            'timestamp': details['timestamp'],
            'reasoning': self._format_reasoning(details),
            'risk_assessment': self._assess_risk(details),
            'position_size': self._calculate_position_size(confidence),
            'agent_breakdown': details['signals']
        }
        
        return decision
    
    def _format_reasoning(self, details: dict) -> str:
        """Format reasoning from all agents"""
        reasoning_parts = []
        for agent_id, signal in details['signals'].items():
            if signal['confidence'] > 0.3:  # Only include confident signals
                reasoning_parts.append(f"{agent_id}: {signal['reasoning']}")
        
        return "; ".join(reasoning_parts)
    
    def _assess_risk(self, details: dict) -> str:
        """Assess overall risk based on agent agreement"""
        actions = [signal['action'] for signal in details['signals'].values()]
        
        # Check agreement
        buy_votes = actions.count(1)
        sell_votes = actions.count(2)
        hold_votes = actions.count(0)
        
        total_votes = len(actions)
        max_votes = max(buy_votes, sell_votes, hold_votes)
        agreement = max_votes / total_votes
        
        if agreement > 0.8:
            return "LOW"  # High agreement = low risk
        elif agreement > 0.6:
            return "MEDIUM"
        else:
            return "HIGH"  # Low agreement = high risk
    
    def _calculate_position_size(self, confidence: float) -> float:
        """Calculate position size based on confidence"""
        # Conservative position sizing
        base_size = 0.1  # 10% of portfolio
        confidence_multiplier = min(confidence * 2, 1.0)  # Max 2x for high confidence
        
        return base_size * confidence_multiplier
    
    def update_performance(self, actual_return: float):
        """Update ensemble performance with actual results"""
        if self.ensemble:
            self.ensemble.update_agent_performance(actual_return)
    
    def get_system_status(self) -> dict:
        """Get comprehensive system status"""
        if not self.ensemble:
            return {"status": "Not initialized"}
        
        performance = self.ensemble.get_performance_summary()
        
        return {
            "status": "Active",
            "symbol": self.symbol,
            "agent_count": len(self.ensemble.agents),
            "total_decisions": len(self.ensemble.decision_history),
            "agent_performance": performance,
            "last_decision": self.ensemble.decision_history[-1] if self.ensemble.decision_history else None
        }

# Example usage
def main():
    # Initialize live trading system
    live_system = LiveTradingSystem("GERN")
    
    # Initialize ensemble (with or without pipeline model)
    ensemble = live_system.initialize_ensemble()
    
    # Example market data
    market_data = {
        'price': 1.45,
        'sma_5': 1.44,
        'sma_10': 1.43,
        'sma_20': 1.42,
        'sma_50': 1.40,
        'sma_200': 1.35,
        'rsi_14': 65,
        'volume': 15000,
        'avg_volume_20': 12000,
        'yearly_high': 1.60,
        'yearly_low': 1.20,
        'intraday_high': 1.46,
        'intraday_low': 1.43,
        'price_1h_ago': 1.44,
        'price_4h_ago': 1.43
    }
    
    # Make trading decision
    decision = live_system.make_live_trading_decision(market_data)
    
    print("Live Trading Decision:")
    print(f"Action: {decision['action']}")
    print(f"Confidence: {decision['confidence']:.2%}")
    print(f"Position Size: {decision['position_size']:.1%}")
    print(f"Risk: {decision['risk_assessment']}")
    print(f"Reasoning: {decision['reasoning']}")
    
    # Simulate performance update
    actual_return = 0.02
    live_system.update_performance(actual_return)
    
    # Get system status
    status = live_system.get_system_status()
    print(f"\nSystem Status: {status['status']}")

if __name__ == "__main__":
    main()

