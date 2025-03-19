# Reinforcement Learning Framework

The TravelGraph system incorporates a comprehensive reinforcement learning (RL) framework to continuously improve agent performance based on user interactions and feedback.

## Overview

The RL system is built to optimize each agent's decision-making processes by learning from:

1. **Explicit feedback**: Direct ratings and comments from users
2. **Implicit feedback**: User interactions and behavior patterns
3. **System metrics**: Success rates, efficiency measures, and error rates

## Architecture

![RL Framework Architecture](images/rl_architecture.png)

The RL framework consists of several key components:

### Experience Collection

- **Experience Buffer**: Stores state-action-reward-next_state tuples for each agent
- **Reward Functions**: Agent-specific reward calculations based on performance metrics
- **State Feature Extraction**: Converts complex system state into ML-friendly features

### Policy Optimization

- **Agent Policies**: Rule-based policies with tunable parameters
- **Training Logic**: Batch processing of experiences to update policies
- **Policy Storage**: Versioned storage of trained policies

### Reward Functions

Each agent has specialized reward functions:

1. **User Interaction Agent**:
   - User satisfaction scores
   - Successful intent understanding
   - Conversation efficiency

2. **Planning Agent**:
   - Preference matching score
   - Itinerary diversity and completeness
   - Budget adherence

3. **Booking Agent**:
   - Booking success rate
   - Price optimization
   - Speed of confirmation

4. **Monitoring Agent**:
   - Alert accuracy
   - Issue resolution rate
   - Proactive monitoring effectiveness

## Implementation

The RL system is implemented in the `reinforcement_learning.py` file and consists of several classes:

### Experience

Stores a single experience tuple:
```python
Experience(
    state_repr=state,          # Dictionary representing the state
    action_name=action_name,   # String identifier for the action
    action_params=params,      # Parameters used for the action
    reward=reward,             # Numerical reward value
    next_state_repr=next_state,# Dictionary representing the next state
    done=done                  # Boolean indicating if episode is done
)
```

### ExperienceBuffer

Manages collections of experiences for each agent:
```python
buffer = ExperienceBuffer(buffer_size=100000)
buffer.add_experience("planning_agent", experience)
batch = buffer.sample_batch("planning_agent", 64)
```

### RewardFunction

Contains static methods for calculating rewards for different agents:
```python
# Example: Calculate planning agent reward
reward = RewardFunction.calculate_itinerary_quality(
    itinerary=trip_itinerary,
    user_preferences=user_prefs
)
```

### AgentPolicy & RuleBasedPolicy

Implements decision logic with tunable parameters:
```python
policy = RuleBasedPolicy("booking_agent")
action_name, params = policy.select_action(state_features)
```

### RLTrainer

Manages the training process:
```python
trainer = RLTrainer(experience_buffer, policies, training_config)
training_results = trainer.train_agent("planning_agent")
```

### RLController

Main controller that coordinates the entire RL system:
```python
controller = RLController("./rl_config.json")
action, params = controller.select_agent_action(agent_name, state)
controller.record_experience(agent_name, next_state, reward_data)
```

## Configuration

The RL system is configured through `rl_config.json`:

```json
{
    "buffer_size": 100000,
    "training": {
        "batch_size": 64,
        "training_frequency": 100,
        "min_experiences": 500,
        "save_frequency": 10
    },
    "reward_weights": {
        "user_satisfaction": 1.0,
        "itinerary_quality": 0.8,
        "booking_efficiency": 0.7,
        "monitoring_effectiveness": 0.6
    },
    "agent_parameters": {
        "user_interaction_agent": {
            "intent_confidence_threshold": 0.7
        }
    }
}
```

## Usage in the System

The RL framework is integrated into the travel planning system through the `RLEnhancedTravelSystem` class, which wraps the base system:

```python
from travel_agent_architecture import TravelPlanningSystem
from reinforcement_learning import RLEnhancedTravelSystem

# Create base system
base_system = TravelPlanningSystem()

# Enhance with RL capabilities
rl_system = RLEnhancedTravelSystem(base_system, "rl_config.json")

# Process a request with RL-optimized parameters
result = rl_system.process_user_request(
    user_id="user123",
    user_input="I want to plan a trip to Paris next month"
)

# Provide explicit feedback
rl_system.provide_explicit_feedback("user123", {
    "satisfaction_score": 4.5,
    "itinerary_rating": 4.8,
    "comments": "Excellent suggestions!"
})
```

## Performance Metrics

The RL system tracks various performance metrics:

- **Average Reward**: Mean reward value for each agent
- **Training Iterations**: Number of policy updates
- **Improvement Rate**: Rate of performance improvement over time
- **Experience Counts**: Number of experiences collected for each agent

These metrics can be viewed in the Grafana dashboard.