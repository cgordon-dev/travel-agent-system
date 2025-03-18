# TravelGraph: Multi-Agent Travel Planning System with Reinforcement Learning

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/langgraph-0.0.33%2B-orange)](https://github.com/langchain-ai/langgraph)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A production-ready AI architecture for autonomous travel planning, combining specialized agents, LangGraph orchestration, and PydanticAI data validation with a continuous learning reinforcement learning framework.

## 🌟 Key Features

- **Multi-Agent Architecture**: Specialized agents collaborate to handle different aspects of travel planning
- **Continuous Learning**: RL framework optimizes agent performance over time based on feedback
- **Structured Knowledge**: PydanticAI ensures type safety and valid data across the entire system
- **Production Ready**: Full integration with web services, data storage, and monitoring
- **Privacy Focused**: Built-in compliance with data privacy regulations

## 📋 System Overview

TravelGraph combines four specialized agents to provide end-to-end travel planning services:

| Agent | Responsibility | Learning Focus |
|-------|---------------|---------------|
| **User Interaction Agent** | Understands user requests and manages conversations | Conversation flow optimization |
| **Planning Agent** | Creates and modifies travel itineraries | Preference matching, activity selection |
| **Booking Agent** | Manages reservations across travel services | Price optimization, booking success |
| **Monitoring Agent** | Tracks price changes and booking status | Alert prioritization, issue detection |

All agents share a centralized team memory and communicate through strongly-typed message passing with LangGraph orchestrating their workflow.

## 🏗️ Architecture Components

### Multi-Agent Communication Framework

- **Message Passing Protocol**: Structured agent communication via the `AgentMessage` model
- **Dynamic Workflow**: LangGraph enables conditional branching, looping, and feedback paths
- **Shared Knowledge**: Team memory with access to all relevant context across agents

### Data Validation System

- **Strongly-Typed Models**: Comprehensive Pydantic models for all data structures
- **Dynamic Schema Evolution**: Models adapt as agents learn new preferences and patterns
- **Validation Guards**: Built-in validation prevents hallucination and data corruption

### Reinforcement Learning Framework

![RL Architecture](https://github.com/yourusername/travel-agent-team/raw/main/docs/images/rl_architecture.png)

The RL system provides continuous improvement through:

1. **Experience Collection**
   - Captures state-action-reward-next_state tuples
   - Persistent storage with efficient memory management
   - Handles both implicit & explicit feedback

2. **Agent-Specific Reward Functions**
   - User satisfaction metrics for interaction quality
   - Preference matching for itinerary planning
   - Booking efficiency & cost optimization 
   - Monitoring effectiveness for detecting issues

3. **Policy Optimization Mechanisms**
   - Parameter tuning based on performance
   - Cross-user learning for generalization
   - A/B testing of agent strategies

### Production Infrastructure

- **Configurable System**: JSON-based configuration for all parameters
- **Persistence Layer**: Stores experiences, models, and user profiles
- **Web Service Integration**: Ready for deployment with production APIs
- **Monitoring & Analytics**: Performance tracking and system insights
- **Security Features**: Data encryption and privacy compliance

## 📂 Project Structure

```
travel-agent-team/
├── travel_agent_architecture.py  # Core multi-agent system
├── pydantic_models.py            # Data validation models
├── reinforcement_learning.py     # RL framework components
├── integration_example.py        # Production integration example
├── langgraph_workflow.py         # Workflow visualization
├── example_usage.py              # Usage examples
├── requirements.txt              # Dependencies
└── rl_config.json                # Configuration file
```

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/travel-agent-team.git
cd travel-agent-team

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from travel_agent_architecture import TravelPlanningSystem

# Initialize the system
system = TravelPlanningSystem()

# Process a user request
result = system.process_user_request(
    user_id="user123",
    user_input="I want to plan a weekend trip to Barcelona in March. We love food tours and need a hotel near the beach."
)

print(f"Itinerary: {result['itinerary']['title']}")
print(f"Total cost: ${result['itinerary']['total_cost']}")
```

### With Reinforcement Learning

```python
from reinforcement_learning import RLEnhancedTravelSystem
from travel_agent_architecture import TravelPlanningSystem

# Initialize base system
base_system = TravelPlanningSystem()

# Enhance with RL capabilities
rl_system = RLEnhancedTravelSystem(base_system, "rl_config.json")

# Process user request with RL enhancement
result = rl_system.process_user_request(
    user_id="user123",
    user_input="I need a business trip to Tokyo next month with meetings in Shinjuku."
)

# Provide explicit feedback
rl_system.provide_explicit_feedback("user123", {
    "satisfaction_score": 4.5,
    "itinerary_rating": 4.8,
    "booking_experience": 4.2,
    "comments": "Great itinerary but booking process could be smoother"
})

# View performance metrics
metrics = rl_system.get_performance_metrics()
print(metrics)
```

### Production Deployment

```python
from integration_example import ProductionSystemServer

# Create production server
server = ProductionSystemServer("production_config.json")

# Start server (in real deployment, use a production WSGI/ASGI server)
server.start(port=8000)
```

## ⚙️ Configuration

TravelGraph is highly configurable through the `rl_config.json` file:

```json
{
    "buffer_size": 100000,
    "training": {
        "batch_size": 64,
        "training_frequency": 100,
        "min_experiences": 500
    },
    "reward_weights": {
        "user_satisfaction": 1.0,
        "itinerary_quality": 0.8,
        "booking_efficiency": 0.7
    },
    "agent_parameters": {
        "user_interaction_agent": {
            "intent_confidence_threshold": 0.7
        }
    },
    "production": {
        "feedback_collection": true,
        "feedback_prompt_frequency": 3
    }
}
```

## 📊 System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                       User Interface                            │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Production Travel System                     │
│  ┌───────────────┐   ┌───────────────┐   ┌───────────────────┐  │
│  │ User Database │   │  RL Controller │   │ Performance Monitor │
│  └───────┬───────┘   └───────┬───────┘   └─────────┬─────────┘  │
│          │                   │                     │            │
└──────────┼───────────────────┼─────────────────────┼────────────┘
           │                   │                     │
           ▼                   ▼                     ▼
┌──────────────────────────────────────────────────────────────────┐
│                    RL-Enhanced Travel System                      │
│  ┌────────────────┐  ┌───────────────┐  ┌────────────────────┐   │
│  │Experience Buffer│  │ Reward Function│  │Policy Optimization│   │
│  └────────┬───────┘  └───────┬───────┘  └─────────┬──────────┘   │
│           │                  │                    │               │
└───────────┼──────────────────┼────────────────────┼───────────────┘
            │                  │                    │
            ▼                  ▼                    ▼
┌───────────────────────────────────────────────────────────────────┐
│                       Base Travel System                           │
│  ┌─────────────────┐    ┌─────────────┐    ┌─────────────────┐    │
│  │User Interaction │    │  Planning   │    │    Booking      │    │
│  │     Agent       │───▶│    Agent    │───▶│     Agent       │    │
│  └─────────────────┘    └─────────────┘    └────────┬────────┘    │
│                                                     │              │
│                                                     ▼              │
│                                            ┌─────────────────┐    │
│                                            │   Monitoring    │    │
│                                            │     Agent       │    │
│                                            └─────────────────┘    │
│                         ┌─────────────┐                           │
│                         │ Team Memory │                           │
│                         └─────────────┘                           │
└───────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌───────────────────────────────────────────────────────────────────┐
│                       External Services                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ ┌─────────────┐│
│  │   Amadeus   │  │    Sabre    │  │ Rentalcars  │ │  KaibanJS   ││
│  └─────────────┘  └─────────────┘  └─────────────┘ └─────────────┘│
└───────────────────────────────────────────────────────────────────┘
```

## 🧠 Learning Mechanisms

### User Preference Learning

The system continuously learns user preferences in several ways:

1. **Explicit Preferences**: Direct statements about preferences
2. **Implicit Preferences**: Choices and reactions to options
3. **Dynamic Schema Evolution**: Adds new preference types as discovered
4. **Cross-Session Learning**: Applies learnings over time

### Agent Parameter Optimization

Each agent's behavior is controlled by parameters that are optimized through RL:

```python
# Parameters tuned through reinforcement learning
parameters = {
    "intent_confidence_threshold": 0.7,  # When to request clarification
    "preference_weight": 0.8,           # Importance of matching preferences
    "budget_weight": 0.7,               # Importance of budget constraints
    "price_alert_threshold": 0.1        # % change to trigger price alerts
}
```

The RL system systematically explores parameter variations and reinforces those that lead to better outcomes.

## 🔧 Extending The System

### Adding New Agent Types

1. Create a new agent class inheriting from the base `Agent` class
2. Define its reward function in `RewardFunction`
3. Add policy logic in `RuleBasedPolicy`
4. Register the agent in the LangGraph workflow

### Creating Custom Reward Functions

```python
def calculate_custom_reward(data: Dict[str, Any]) -> float:
    """Calculate custom reward based on specific metrics"""
    score = 0.0
    
    # Add reward logic based on your specific domain
    if data.get("success", False):
        score += 1.0
    
    # Add penalties for issues
    if data.get("errors", 0) > 0:
        score -= 0.5
        
    return max(0.0, min(score, 2.0))  # Clamp between 0 and 2
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this code in your research, please cite:

```
@software{travelgraph2023,
  author = {Your Name},
  title = {TravelGraph: Multi-Agent Travel Planning System with Reinforcement Learning},
  year = {2023},
  url = {https://github.com/yourusername/travel-agent-team}
}
```

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for LangGraph
- [Pydantic](https://github.com/pydantic/pydantic) for data validation
- The reinforcement learning community for algorithms and best practices