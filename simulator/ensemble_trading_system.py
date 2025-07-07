import numpy as np
import torch
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class TradingSignal:
    """Standardized trading signal from any agent"""
    agent_id: str
    action: int  # 0=hold, 1=buy, 2=sell
    confidence: float  # 0.0 to 1.0
    reasoning: str
    timestamp: datetime
    market_context: Dict[str, float]

@dataclass
class AgentConfig:
    """Configuration for each specialized agent"""
    agent_id: str
    training_period: str  # "3_years", "1_year", "3_months", "1_week", "pipeline"
    weight: float  # Voting weight in ensemble
    data_lookback_days: int
    retrain_frequency: str  # "never", "weekly", "monthly", "quarterly"
    specialization: str  # "long_term", "medium_term", "short_term", "adaptive"

class BaseAgent(ABC):
    """Base class for all trading agents"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.model = None
        self.last_retrain = None
        self.performance_history = []
        
    @abstractmethod
    def generate_signal(self, market_state: Dict[str, float]) -> TradingSignal:
        """Generate trading signal based on market state"""
        pass
    
    @abstractmethod
    def update_performance(self, actual_return: float, predicted_action: int):
        """Update agent's performance tracking"""
        pass

class LongTermAgent(BaseAgent):
    """Agent trained on 3 years of data - focuses on major trends"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.trend_memory = []
        self.major_support_resistance = {}
        
    def generate_signal(self, market_state: Dict[str, float]) -> TradingSignal:
        """Generate signal based on long-term trends and patterns"""
        
        # Analyze long-term trend
        price = market_state.get('price', 0)
        sma_200 = market_state.get('sma_200', price)
        yearly_high = market_state.get('yearly_high', price)
        yearly_low = market_state.get('yearly_low', price)
        
        # Long-term trend analysis
        trend_strength = (price - sma_200) / sma_200 if sma_200 > 0 else 0
        position_in_range = (price - yearly_low) / (yearly_high - yearly_low) if yearly_high > yearly_low else 0.5
        
        # Decision logic for long-term perspective
        if trend_strength > 0.1 and position_in_range < 0.3:  # Strong uptrend, near yearly low
            action = 1  # BUY
            confidence = min(0.8, abs(trend_strength) + (0.5 - position_in_range))
            reasoning = f"Long-term uptrend ({trend_strength:.2%}), near yearly low ({position_in_range:.1%})"
        elif trend_strength < -0.1 and position_in_range > 0.7:  # Strong downtrend, near yearly high
            action = 2  # SELL
            confidence = min(0.8, abs(trend_strength) + (position_in_range - 0.5))
            reasoning = f"Long-term downtrend ({trend_strength:.2%}), near yearly high ({position_in_range:.1%})"
        else:
            action = 0  # HOLD
            confidence = 0.3
            reasoning = "No clear long-term signal"
        
        return TradingSignal(
            agent_id=self.config.agent_id,
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            timestamp=datetime.now(),
            market_context={'trend_strength': trend_strength, 'position_in_range': position_in_range}
        )
    
    def update_performance(self, actual_return: float, predicted_action: int):
        """Track long-term prediction accuracy"""
        self.performance_history.append({
            'timestamp': datetime.now(),
            'predicted_action': predicted_action,
            'actual_return': actual_return,
            'correct': (predicted_action == 1 and actual_return > 0.02) or 
                     (predicted_action == 2 and actual_return < -0.02) or
                     (predicted_action == 0 and abs(actual_return) < 0.02)
        })

class MediumTermAgent(BaseAgent):
    """Agent trained on 1 year of data - focuses on seasonal patterns"""
    
    def generate_signal(self, market_state: Dict[str, float]) -> TradingSignal:
        """Generate signal based on medium-term patterns"""
        
        price = market_state.get('price', 0)
        sma_50 = market_state.get('sma_50', price)
        sma_20 = market_state.get('sma_20', price)
        rsi = market_state.get('rsi_14', 50)
        
        # Medium-term momentum
        momentum_50 = (price - sma_50) / sma_50 if sma_50 > 0 else 0
        momentum_20 = (price - sma_20) / sma_20 if sma_20 > 0 else 0
        
        # RSI analysis
        rsi_signal = 0
        if rsi < 30:  # Oversold
            rsi_signal = 1
        elif rsi > 70:  # Overbought
            rsi_signal = -1
        
        # Decision logic
        if momentum_50 > 0.05 and momentum_20 > 0.02 and rsi_signal >= 0:
            action = 1  # BUY
            confidence = min(0.7, momentum_50 + momentum_20)
            reasoning = f"Medium-term momentum positive, RSI favorable"
        elif momentum_50 < -0.05 and momentum_20 < -0.02 and rsi_signal <= 0:
            action = 2  # SELL
            confidence = min(0.7, abs(momentum_50) + abs(momentum_20))
            reasoning = f"Medium-term momentum negative, RSI unfavorable"
        else:
            action = 0  # HOLD
            confidence = 0.4
            reasoning = "Mixed medium-term signals"
        
        return TradingSignal(
            agent_id=self.config.agent_id,
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            timestamp=datetime.now(),
            market_context={'momentum_50': momentum_50, 'momentum_20': momentum_20, 'rsi': rsi}
        )
    
    def update_performance(self, actual_return: float, predicted_action: int):
        """Track medium-term prediction accuracy"""
        self.performance_history.append({
            'timestamp': datetime.now(),
            'predicted_action': predicted_action,
            'actual_return': actual_return
        })

class ShortTermAgent(BaseAgent):
    """Agent trained on 3 months of data - focuses on recent patterns"""
    
    def generate_signal(self, market_state: Dict[str, float]) -> TradingSignal:
        """Generate signal based on short-term patterns"""
        
        price = market_state.get('price', 0)
        sma_5 = market_state.get('sma_5', price)
        sma_10 = market_state.get('sma_10', price)
        volume = market_state.get('volume', 1000)
        avg_volume = market_state.get('avg_volume_20', volume)
        
        # Short-term momentum
        momentum_5 = (price - sma_5) / sma_5 if sma_5 > 0 else 0
        momentum_10 = (price - sma_10) / sma_10 if sma_10 > 0 else 0
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        
        # Decision logic for short-term trading
        if momentum_5 > 0.02 and momentum_10 > 0.01 and volume_ratio > 1.5:
            action = 1  # BUY
            confidence = min(0.6, momentum_5 * 2 + (volume_ratio - 1) * 0.2)
            reasoning = f"Short-term breakout with volume confirmation"
        elif momentum_5 < -0.02 and momentum_10 < -0.01 and volume_ratio > 1.2:
            action = 2  # SELL
            confidence = min(0.6, abs(momentum_5) * 2 + (volume_ratio - 1) * 0.2)
            reasoning = f"Short-term breakdown with volume"
        else:
            action = 0  # HOLD
            confidence = 0.2
            reasoning = "No clear short-term signal"
        
        return TradingSignal(
            agent_id=self.config.agent_id,
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            timestamp=datetime.now(),
            market_context={'momentum_5': momentum_5, 'volume_ratio': volume_ratio}
        )
    
    def update_performance(self, actual_return: float, predicted_action: int):
        """Track short-term prediction accuracy"""
        self.performance_history.append({
            'timestamp': datetime.now(),
            'predicted_action': predicted_action,
            'actual_return': actual_return
        })

class WeeklyAgent(BaseAgent):
    """Agent trained on 1 week of data - focuses on intraday patterns"""
    
    def generate_signal(self, market_state: Dict[str, float]) -> TradingSignal:
        """Generate signal based on very recent patterns"""
        
        price = market_state.get('price', 0)
        price_1h_ago = market_state.get('price_1h_ago', price)
        price_4h_ago = market_state.get('price_4h_ago', price)
        intraday_high = market_state.get('intraday_high', price)
        intraday_low = market_state.get('intraday_low', price)
        
        # Very short-term momentum
        momentum_1h = (price - price_1h_ago) / price_1h_ago if price_1h_ago > 0 else 0
        momentum_4h = (price - price_4h_ago) / price_4h_ago if price_4h_ago > 0 else 0
        
        # Position within intraday range
        if intraday_high > intraday_low:
            intraday_position = (price - intraday_low) / (intraday_high - intraday_low)
        else:
            intraday_position = 0.5
        
        # Decision logic for intraday trading
        if momentum_1h > 0.005 and momentum_4h > 0.002 and intraday_position < 0.8:
            action = 1  # BUY
            confidence = min(0.5, momentum_1h * 10)
            reasoning = f"Intraday momentum building, room to move up"
        elif momentum_1h < -0.005 and momentum_4h < -0.002 and intraday_position > 0.2:
            action = 2  # SELL
            confidence = min(0.5, abs(momentum_1h) * 10)
            reasoning = f"Intraday momentum declining, room to move down"
        else:
            action = 0  # HOLD
            confidence = 0.1
            reasoning = "No clear intraday signal"
        
        return TradingSignal(
            agent_id=self.config.agent_id,
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            timestamp=datetime.now(),
            market_context={'momentum_1h': momentum_1h, 'intraday_position': intraday_position}
        )
    
    def update_performance(self, actual_return: float, predicted_action: int):
        """Track intraday prediction accuracy"""
        self.performance_history.append({
            'timestamp': datetime.now(),
            'predicted_action': predicted_action,
            'actual_return': actual_return
        })

class PipelineAgent(BaseAgent):
    """Agent from your pipeline - adaptive and continuously learning"""
    
    def __init__(self, config: AgentConfig, pipeline_model=None):
        super().__init__(config)
        self.pipeline_model = pipeline_model
        self.pattern_explorer = None
        
    def generate_signal(self, market_state: Dict[str, float]) -> TradingSignal:
        """Generate signal using pipeline-trained model"""
        
        if self.pipeline_model is None:
            return TradingSignal(
                agent_id=self.config.agent_id,
                action=0,
                confidence=0.0,
                reasoning="Pipeline model not loaded",
                timestamp=datetime.now(),
                market_context={}
            )
        
        # Convert market_state to model input format
        state_vector = self._prepare_state_vector(market_state)
        
        # Get model prediction
        with torch.no_grad():
            q_values = self.pipeline_model(torch.FloatTensor(state_vector).unsqueeze(0))
            action = torch.argmax(q_values).item()
            confidence = torch.softmax(q_values, dim=1).max().item()
        
        reasoning = f"Pipeline model prediction (confidence: {confidence:.2f})"
        
        return TradingSignal(
            agent_id=self.config.agent_id,
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            timestamp=datetime.now(),
            market_context={'q_values': q_values.numpy().tolist()}
        )
    
    def _prepare_state_vector(self, market_state: Dict[str, float]) -> np.ndarray:
        """Convert market state to model input format"""
        # This should match your strategy_optimizer.py state format
        price = market_state.get('price', 0)
        sma_5 = market_state.get('sma_5', price)
        sma_10 = market_state.get('sma_10', price)
        sma_20 = market_state.get('sma_20', price)
        rsi_14 = market_state.get('rsi_14', 50)
        
        base_state = np.array([
            price/100, sma_5/100, sma_10/100, sma_20/100, rsi_14/100,
            0, 1.0, 0  # position, cash_ratio, holdings_ratio
        ], dtype=np.float32)
        
        # Add pattern guidance (simplified)
        guidance = np.array([0, 0, 0], dtype=np.float32)  # Placeholder
        
        return np.concatenate([base_state, guidance])
    
    def update_performance(self, actual_return: float, predicted_action: int):
        """Track pipeline model performance"""
        self.performance_history.append({
            'timestamp': datetime.now(),
            'predicted_action': predicted_action,
            'actual_return': actual_return
        })

class EnsembleDecisionMaker:
    """Main agent that aggregates signals from all specialized agents"""
    
    def __init__(self, agents: List[BaseAgent], weights: Dict[str, float]):
        self.agents = agents
        self.weights = weights
        self.decision_history = []
        self.performance_tracker = {}
        
        # Initialize performance tracking for each agent
        for agent in agents:
            self.performance_tracker[agent.config.agent_id] = {
                'recent_accuracy': 0.5,
                'confidence_calibration': 1.0,
                'dynamic_weight': weights.get(agent.config.agent_id, 0.2)
            }
    
    def make_trading_decision(self, market_state: Dict[str, float]) -> Tuple[int, float, Dict]:
        """Aggregate signals from all agents and make final decision"""
        
        # Collect signals from all agents
        signals = []
        for agent in self.agents:
            try:
                signal = agent.generate_signal(market_state)
                signals.append(signal)
            except Exception as e:
                print(f"Error getting signal from {agent.config.agent_id}: {e}")
                continue
        
        if not signals:
            return 0, 0.0, {"error": "No signals received"}
        
        # Calculate weighted votes
        action_votes = {0: 0.0, 1: 0.0, 2: 0.0}  # hold, buy, sell
        total_weight = 0.0
        signal_details = {}
        
        for signal in signals:
            agent_id = signal.agent_id
            base_weight = self.weights.get(agent_id, 0.2)
            
            # Adjust weight based on recent performance
            performance = self.performance_tracker.get(agent_id, {})
            dynamic_weight = base_weight * performance.get('confidence_calibration', 1.0)
            
            # Weight the vote by confidence and performance
            weighted_vote = dynamic_weight * signal.confidence
            action_votes[signal.action] += weighted_vote
            total_weight += dynamic_weight
            
            signal_details[agent_id] = {
                'action': signal.action,
                'confidence': signal.confidence,
                'weight': dynamic_weight,
                'reasoning': signal.reasoning,
                'market_context': signal.market_context
            }
        
        # Determine final action
        if total_weight > 0:
            # Normalize votes
            for action in action_votes:
                action_votes[action] /= total_weight
            
            # Find winning action
            final_action = max(action_votes, key=action_votes.get)
            final_confidence = action_votes[final_action]
            
            # Require minimum confidence threshold
            min_confidence_threshold = 0.6
            if final_confidence < min_confidence_threshold:
                final_action = 0  # Default to HOLD if not confident enough
                final_confidence = action_votes[0]
        else:
            final_action = 0
            final_confidence = 0.0
        
        # Record decision
        decision_record = {
            'timestamp': datetime.now(),
            'final_action': final_action,
            'final_confidence': final_confidence,
            'action_votes': action_votes,
            'signals': signal_details,
            'market_state': market_state
        }
        
        self.decision_history.append(decision_record)
        
        return final_action, final_confidence, decision_record
    
    def update_agent_performance(self, actual_return: float, time_horizon: str = "1h"):
        """Update performance tracking for all agents based on actual results"""
        
        if not self.decision_history:
            return
        
        # Find relevant decision based on time horizon
        cutoff_time = datetime.now() - timedelta(hours=1 if time_horizon == "1h" else 24)
        relevant_decisions = [d for d in self.decision_history if d['timestamp'] > cutoff_time]
        
        for decision in relevant_decisions:
            for agent_id, signal_info in decision['signals'].items():
                predicted_action = signal_info['action']
                
                # Update individual agent performance
                agent = next((a for a in self.agents if a.config.agent_id == agent_id), None)
                if agent:
                    agent.update_performance(actual_return, predicted_action)
                
                # Update ensemble performance tracking
                if agent_id in self.performance_tracker:
                    self._update_performance_metrics(agent_id, actual_return, predicted_action)
    
    def _update_performance_metrics(self, agent_id: str, actual_return: float, predicted_action: int):
        """Update performance metrics for dynamic weight adjustment"""
        
        # Determine if prediction was correct
        correct = False
        if predicted_action == 1 and actual_return > 0.01:  # Buy was correct
            correct = True
        elif predicted_action == 2 and actual_return < -0.01:  # Sell was correct
            correct = True
        elif predicted_action == 0 and abs(actual_return) < 0.01:  # Hold was correct
            correct = True
        
        # Update recent accuracy with exponential moving average
        tracker = self.performance_tracker[agent_id]
        alpha = 0.1  # Learning rate
        tracker['recent_accuracy'] = (1 - alpha) * tracker['recent_accuracy'] + alpha * (1.0 if correct else 0.0)
        
        # Adjust confidence calibration
        if tracker['recent_accuracy'] > 0.6:
            tracker['confidence_calibration'] = min(1.5, tracker['confidence_calibration'] + 0.05)
        elif tracker['recent_accuracy'] < 0.4:
            tracker['confidence_calibration'] = max(0.5, tracker['confidence_calibration'] - 0.05)
        
        # Update dynamic weight
        base_weight = self.weights[agent_id]
        performance_multiplier = tracker['recent_accuracy'] * tracker['confidence_calibration']
        tracker['dynamic_weight'] = base_weight * performance_multiplier
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary for all agents"""
        summary = {}
        for agent_id, tracker in self.performance_tracker.items():
            agent = next((a for a in self.agents if a.config.agent_id == agent_id), None)
            if agent and agent.performance_history:
                recent_performance = agent.performance_history[-10:] if len(agent.performance_history) >= 10 else agent.performance_history
                
                summary[agent_id] = {
                    'base_weight': self.weights[agent_id],
                    'dynamic_weight': tracker['dynamic_weight'],
                    'recent_accuracy': tracker['recent_accuracy'],
                    'confidence_calibration': tracker['confidence_calibration'],
                    'total_predictions': len(agent.performance_history),
                    'recent_predictions': len(recent_performance)
                }
        
        return summary

def create_ensemble_system(pipeline_model=None) -> EnsembleDecisionMaker:
    """Factory function to create the complete ensemble system"""
    
    # Define agent configurations
    agent_configs = [
        AgentConfig(
            agent_id="long_term_3y",
            training_period="3_years",
            weight=0.25,
            data_lookback_days=1095,
            retrain_frequency="quarterly",
            specialization="long_term"
        ),
        AgentConfig(
            agent_id="medium_term_1y",
            training_period="1_year",
            weight=0.20,
            data_lookback_days=365,
            retrain_frequency="monthly",
            specialization="medium_term"
        ),
        AgentConfig(
            agent_id="short_term_3m",
            training_period="3_months",
            weight=0.20,
            data_lookback_days=90,
            retrain_frequency="weekly",
            specialization="short_term"
        ),
        AgentConfig(
            agent_id="weekly_agent",
            training_period="1_week",
            weight=0.10,
            data_lookback_days=7,
            retrain_frequency="daily",
            specialization="intraday"
        ),
        AgentConfig(
            agent_id="pipeline_agent",
            training_period="pipeline",
            weight=0.25,
            data_lookback_days=60,
            retrain_frequency="weekly",
            specialization="adaptive"
        )
    ]
    
    # Create specialized agents
    agents = [
        LongTermAgent(agent_configs[0]),
        MediumTermAgent(agent_configs[1]),
        ShortTermAgent(agent_configs[2]),
        WeeklyAgent(agent_configs[3]),
        PipelineAgent(agent_configs[4], pipeline_model)
    ]
    
    # Define weights
    weights = {config.agent_id: config.weight for config in agent_configs}
    
    # Create ensemble decision maker
    ensemble = EnsembleDecisionMaker(agents, weights)
    
    return ensemble

# Example usage
def main():
    """Example of how to use the ensemble system"""
    
    # Create ensemble system
    ensemble = create_ensemble_system()
    
    # Example market state
    market_state = {
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
    action, confidence, details = ensemble.make_trading_decision(market_state)
    
    print(f"Final Decision: {['HOLD', 'BUY', 'SELL'][action]}")
    print(f"Confidence: {confidence:.2%}")
    print("\nAgent Signals:")
    for agent_id, signal in details['signals'].items():
        print(f"  {agent_id}: {['HOLD', 'BUY', 'SELL'][signal['action']]} "
              f"(conf: {signal['confidence']:.2f}, weight: {signal['weight']:.2f})")
        print(f"    Reasoning: {signal['reasoning']}")
    
    # Simulate performance update (after 1 hour)
    actual_return = 0.015  # 1.5% return
    ensemble.update_agent_performance(actual_return, "1h")
    
    # Get performance summary
    performance = ensemble.get_performance_summary()
    print("\nAgent Performance:")
    for agent_id, perf in performance.items():
        print(f"  {agent_id}: accuracy={perf['recent_accuracy']:.2%}, "
              f"weight={perf['base_weight']:.2f}→{perf['dynamic_weight']:.2f}")

if __name__ == "__main__":
    main()

