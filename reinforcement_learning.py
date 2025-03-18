"""
Reinforcement Learning Framework for Travel Agent System
Implements production-ready RL capabilities to improve agent performance over time
"""

import os
import json
import time
import uuid
import numpy as np
import pickle
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import deque
import threading
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rl_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rl_framework")

# Import local modules
from pydantic_models import (
    UserPreference, 
    TravelItinerary, 
    BookingConfirmation, 
    AgentMessage,
    TeamMemory
)

class Experience:
    """
    Stores experience tuples (state, action, reward, next_state) for RL training
    """
    def __init__(
        self, 
        state_repr: Dict[str, Any], 
        action_name: str, 
        action_params: Dict[str, Any],
        reward: float, 
        next_state_repr: Dict[str, Any], 
        done: bool,
        metadata: Dict[str, Any] = None
    ):
        self.state_repr = state_repr
        self.action_name = action_name
        self.action_params = action_params
        self.reward = reward
        self.next_state_repr = next_state_repr
        self.done = done
        self.timestamp = datetime.now()
        self.metadata = metadata or {}
        self.experience_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert experience to dictionary for storage"""
        return {
            "experience_id": self.experience_id,
            "state_repr": self.state_repr,
            "action_name": self.action_name,
            "action_params": self.action_params,
            "reward": self.reward,
            "next_state_repr": self.next_state_repr,
            "done": self.done,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Experience':
        """Create Experience instance from dictionary"""
        experience = cls(
            state_repr=data["state_repr"],
            action_name=data["action_name"],
            action_params=data["action_params"],
            reward=data["reward"],
            next_state_repr=data["next_state_repr"],
            done=data["done"],
            metadata=data.get("metadata", {})
        )
        experience.experience_id = data["experience_id"]
        experience.timestamp = datetime.fromisoformat(data["timestamp"])
        return experience


class ExperienceBuffer:
    """
    Manages a buffer of experiences for each agent with efficient storage
    """
    def __init__(self, buffer_size: int = 100000, storage_path: str = "./rl_data"):
        self.buffer_size = buffer_size
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory buffer for recent experiences
        self.buffer = {
            "user_interaction_agent": deque(maxlen=buffer_size),
            "planning_agent": deque(maxlen=buffer_size),
            "booking_agent": deque(maxlen=buffer_size),
            "monitoring_agent": deque(maxlen=buffer_size)
        }
        
        # Load existing experiences
        self._load_experiences()
        
        # Setup periodic save
        self.save_thread = threading.Thread(target=self._periodic_save, daemon=True)
        self.save_thread.start()
        
        # Experience count tracking
        self.experience_counts = {agent: len(buffer) for agent, buffer in self.buffer.items()}
        logger.info(f"Experience buffer initialized with counts: {self.experience_counts}")
    
    def add_experience(self, agent_name: str, experience: Experience) -> None:
        """Add a new experience to the buffer"""
        if agent_name not in self.buffer:
            self.buffer[agent_name] = deque(maxlen=self.buffer_size)
        
        self.buffer[agent_name].append(experience)
        self.experience_counts[agent_name] = len(self.buffer[agent_name])
        
        # If buffer is getting full, save to disk
        if len(self.buffer[agent_name]) >= self.buffer_size * 0.9:
            self._save_experiences(agent_name)
    
    def sample_batch(self, agent_name: str, batch_size: int) -> List[Experience]:
        """Sample a random batch of experiences for a specific agent"""
        if agent_name not in self.buffer or len(self.buffer[agent_name]) == 0:
            return []
        
        buffer = self.buffer[agent_name]
        indices = np.random.choice(len(buffer), min(batch_size, len(buffer)), replace=False)
        return [buffer[i] for i in indices]
    
    def _save_experiences(self, agent_name: str) -> None:
        """Save experiences to disk"""
        agent_path = self.storage_path / agent_name
        agent_path.mkdir(exist_ok=True)
        
        # Create a batch file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = agent_path / f"experiences_{timestamp}.json"
        
        experiences_data = [exp.to_dict() for exp in self.buffer[agent_name]]
        with open(file_path, 'w') as f:
            json.dump(experiences_data, f)
        
        logger.info(f"Saved {len(experiences_data)} experiences for {agent_name} to {file_path}")
    
    def _load_experiences(self) -> None:
        """Load experiences from disk"""
        for agent_name in self.buffer.keys():
            agent_path = self.storage_path / agent_name
            if not agent_path.exists():
                continue
            
            # Load most recent files up to buffer size
            experience_files = sorted(list(agent_path.glob("experiences_*.json")), reverse=True)
            
            total_loaded = 0
            for file_path in experience_files:
                if total_loaded >= self.buffer_size:
                    break
                
                try:
                    with open(file_path, 'r') as f:
                        experiences_data = json.load(f)
                    
                    # Load experiences up to buffer size
                    for exp_data in experiences_data:
                        if total_loaded >= self.buffer_size:
                            break
                        
                        experience = Experience.from_dict(exp_data)
                        self.buffer[agent_name].append(experience)
                        total_loaded += 1
                
                except Exception as e:
                    logger.error(f"Error loading experiences from {file_path}: {e}")
            
            logger.info(f"Loaded {total_loaded} experiences for {agent_name}")
    
    def _periodic_save(self) -> None:
        """Periodically save experiences to disk"""
        while True:
            time.sleep(300)  # Save every 5 minutes
            for agent_name in self.buffer.keys():
                if len(self.buffer[agent_name]) > 0:
                    self._save_experiences(agent_name)


class RewardFunction:
    """
    Defines reward functions for different agent actions and outcomes
    """
    @staticmethod
    def calculate_user_satisfaction(user_feedback: Optional[Dict[str, Any]] = None) -> float:
        """Calculate reward based on explicit or implicit user feedback"""
        if not user_feedback:
            return 0.0
        
        satisfaction_score = user_feedback.get("satisfaction_score", 0.0)
        
        # Apply modifiers based on feedback aspects
        if user_feedback.get("itinerary_accepted", False):
            satisfaction_score += 1.0
        
        if user_feedback.get("booking_successful", False):
            satisfaction_score += 0.5
        
        if user_feedback.get("price_savings", 0.0) > 0:
            # Normalize savings to a reasonable reward
            savings_percent = min(user_feedback["price_savings"] / 100.0, 0.5)
            satisfaction_score += savings_percent
        
        # Negative rewards for issues
        if user_feedback.get("errors", 0) > 0:
            satisfaction_score -= 0.5 * min(user_feedback["errors"], 3)
        
        if user_feedback.get("booking_failures", 0) > 0:
            satisfaction_score -= 1.0
        
        return max(-1.0, min(satisfaction_score, 2.0))  # Clamp between -1 and 2
    
    @staticmethod
    def calculate_itinerary_quality(itinerary: TravelItinerary, user_preferences: UserPreference) -> float:
        """Calculate reward based on itinerary quality and preference matching"""
        if not itinerary or not user_preferences:
            return 0.0
        
        score = 0.0
        
        # Check if itinerary respects budget constraints
        if user_preferences.budget:
            if itinerary.total_cost <= user_preferences.budget.max_price:
                score += 0.5
                # Bonus for finding good deals within budget
                budget_efficiency = 1.0 - ((itinerary.total_cost - user_preferences.budget.min_price) / 
                                          (user_preferences.budget.max_price - user_preferences.budget.min_price))
                score += max(0, min(budget_efficiency, 0.5))
            else:
                # Penalty for exceeding budget
                score -= min(1.0, (itinerary.total_cost - user_preferences.budget.max_price) / 
                            user_preferences.budget.max_price)
        
        # Check if accommodation types match preferences
        if user_preferences.accommodation_types:
            accommodation_match_ratio = sum(1 for a in itinerary.accommodations 
                                         if a.accommodation_type in user_preferences.accommodation_types) / \
                                    max(1, len(itinerary.accommodations))
            score += accommodation_match_ratio * 0.5
        
        # Check if activity types match preferences
        if user_preferences.activity_types:
            activity_match_ratio = sum(1 for a in itinerary.activities 
                                     if a.activity_type in user_preferences.activity_types) / \
                                max(1, len(itinerary.activities))
            score += activity_match_ratio * 0.5
        
        # Balance of activities - reward diverse itineraries
        activity_types = set(a.activity_type for a in itinerary.activities)
        diversity_score = len(activity_types) / max(1, min(len(ActivityType), len(itinerary.activities)))
        score += diversity_score * 0.2
        
        # Accommodation quality (assuming amenities count correlates with quality)
        avg_amenities = sum(len(a.amenities) for a in itinerary.accommodations) / max(1, len(itinerary.accommodations))
        score += min(avg_amenities / 10, 0.3)  # Cap at 0.3
        
        return max(0.0, min(score, 2.0))  # Clamp between 0 and 2
    
    @staticmethod
    def calculate_booking_efficiency(
        bookings: List[BookingConfirmation],
        itinerary: TravelItinerary
    ) -> float:
        """Calculate reward based on booking efficiency and success rate"""
        if not bookings or not itinerary:
            return 0.0
        
        score = 0.0
        
        # Booking success rate
        required_bookings = len(itinerary.accommodations) + len(itinerary.transportations) + \
                          sum(1 for a in itinerary.activities if getattr(a, 'requires_booking', False))
        
        if required_bookings > 0:
            success_rate = len(bookings) / required_bookings
            score += success_rate
        
        # Speed bonus - faster booking gets higher reward
        if bookings and hasattr(bookings[0], 'booking_timestamp') and hasattr(bookings[0], 'request_timestamp'):
            avg_booking_time = sum((b.booking_timestamp - b.request_timestamp).total_seconds() 
                               for b in bookings if hasattr(b, 'booking_timestamp') and hasattr(b, 'request_timestamp')) / len(bookings)
            # Normalize - assume 60 seconds is very good, 300 seconds is average
            time_score = max(0, min(1, (300 - avg_booking_time) / 240))
            score += time_score * 0.5
        
        # Price efficiency - reward for finding cheaper options
        if hasattr(itinerary, 'original_estimated_cost') and itinerary.original_estimated_cost > 0:
            savings_ratio = max(0, (itinerary.original_estimated_cost - itinerary.total_cost) / itinerary.original_estimated_cost)
            score += savings_ratio * 0.5
        
        return max(0.0, min(score, 2.0))  # Clamp between 0 and 2
    
    @staticmethod
    def calculate_monitoring_effectiveness(
        alerts: List[Dict[str, Any]],
        resolved_issues: List[Dict[str, Any]]
    ) -> float:
        """Calculate reward based on monitoring effectiveness and issue resolution"""
        if not alerts:
            return 0.1  # Small baseline reward for monitoring with no issues
        
        score = 0.0
        
        # Calculate issue detection reward
        if alerts:
            # Each alert gets a small reward
            score += min(len(alerts) * 0.1, 0.5)
            
            # Higher rewards for detecting high-severity issues
            high_severity_count = sum(1 for a in alerts if a.get('severity') in ('high', 'critical'))
            score += high_severity_count * 0.2
        
        # Calculate issue resolution reward
        if resolved_issues:
            resolution_ratio = len(resolved_issues) / max(1, len(alerts))
            score += resolution_ratio
            
            # Bonus for fast resolutions
            if all(('detection_time' in i and 'resolution_time' in i) for i in resolved_issues):
                avg_resolution_time = sum((i['resolution_time'] - i['detection_time']).total_seconds() 
                                     for i in resolved_issues) / len(resolved_issues)
                # Normalize - faster is better
                time_score = max(0, min(1, (3600 - avg_resolution_time) / 3600))  # 1 hour reference
                score += time_score * 0.5
        
        # Penalty for unresolved critical issues
        unresolved_critical = sum(1 for a in alerts if a.get('severity') in ('high', 'critical') 
                                and a.get('id') not in [r.get('alert_id') for r in resolved_issues])
        score -= unresolved_critical * 0.3
        
        return max(0.0, min(score, 2.0))  # Clamp between 0 and 2


class StateFeatureExtractor:
    """
    Extracts relevant features from system state for RL algorithms
    """
    @staticmethod
    def extract_user_features(user_preferences: UserPreference) -> Dict[str, Any]:
        """Extract key features from user preferences"""
        features = {
            "has_budget": user_preferences.budget is not None,
            "budget_range": user_preferences.budget.max_price - user_preferences.budget.min_price if user_preferences.budget else 0,
            "accommodation_preferences": [t for t in user_preferences.accommodation_types],
            "activity_preferences": [t for t in user_preferences.activity_types],
            "accessibility_requirements": len(user_preferences.accessibility_requirements) > 0,
            "dietary_restrictions": len(user_preferences.dietary_restrictions) > 0,
            "travel_history_count": len(user_preferences.traveler_profile.get("previous_destinations", [])),
            "learned_preference_count": len(user_preferences.learned_preferences)
        }
        return features
    
    @staticmethod
    def extract_itinerary_features(itinerary: Optional[TravelItinerary]) -> Dict[str, Any]:
        """Extract key features from travel itinerary"""
        if not itinerary:
            return {
                "has_itinerary": False,
                "duration_days": 0,
                "total_cost": 0,
                "accommodation_count": 0,
                "transportation_count": 0,
                "activity_count": 0,
                "booking_completion": 0
            }
        
        # Calculate what percentage of items have booking references
        total_items = (len(itinerary.accommodations) + len(itinerary.transportations) + 
                      len(itinerary.activities))
        booked_items = sum(1 for items in [itinerary.accommodations, itinerary.transportations, itinerary.activities]
                          for item in items if getattr(item, "booking_reference", None))
        
        booking_completion = booked_items / max(1, total_items)
        
        features = {
            "has_itinerary": True,
            "duration_days": (itinerary.end_date - itinerary.start_date).days,
            "total_cost": itinerary.total_cost,
            "accommodation_count": len(itinerary.accommodations),
            "transportation_count": len(itinerary.transportations),
            "activity_count": len(itinerary.activities),
            "booking_completion": booking_completion
        }
        return features
    
    @staticmethod
    def extract_conversation_features(conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract key features from conversation history"""
        if not conversation_history:
            return {
                "conversation_turns": 0,
                "user_sentiment": 0,
                "query_complexity": 0
            }
        
        # Count conversation turns
        user_turns = sum(1 for msg in conversation_history if msg.get("role") == "user")
        
        # Estimate query complexity based on message length
        user_messages = [msg.get("content", "") for msg in conversation_history if msg.get("role") == "user"]
        avg_message_length = sum(len(msg) for msg in user_messages) / max(1, len(user_messages))
        query_complexity = min(1.0, avg_message_length / 200)  # Normalize to 0-1
        
        features = {
            "conversation_turns": user_turns,
            "query_complexity": query_complexity
        }
        return features
    
    @staticmethod
    def extract_state_features(team_memory: TeamMemory) -> Dict[str, Any]:
        """Extract comprehensive state features from team memory"""
        user_features = StateFeatureExtractor.extract_user_features(team_memory.user_profile)
        itinerary_features = StateFeatureExtractor.extract_itinerary_features(team_memory.current_itinerary)
        conversation_features = StateFeatureExtractor.extract_conversation_features(team_memory.conversation_history)
        
        # Additional system state features
        system_features = {
            "has_alerts": len(team_memory.monitoring_alerts) > 0,
            "alert_count": len(team_memory.monitoring_alerts),
            "has_booking_history": len(team_memory.booking_history) > 0,
            "booking_count": len(team_memory.booking_history)
        }
        
        # Combine all features
        features = {
            **user_features,
            **itinerary_features,
            **conversation_features,
            **system_features,
            "timestamp": datetime.now().timestamp()
        }
        
        return features


class AgentPolicy:
    """
    Base class for agent policies that can be trained with RL
    """
    def __init__(self, agent_name: str, model_dir: str = "./rl_models"):
        self.agent_name = agent_name
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / f"{agent_name}_policy.pkl"
        
        # Policy parameters
        self.parameters = {}
        
        # Load existing policy if available
        self._load_policy()
        
        # Training stats
        self.training_iterations = 0
        self.last_training_time = None
        self.cumulative_reward = 0
        self.version = 1
    
    def select_action(self, state_features: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Select an action based on the current state
        Returns action name and parameters
        """
        # Base implementation uses a simple rule-based approach
        # Subclasses should implement more sophisticated policies
        return "default_action", {}
    
    def update_policy(self, experiences: List[Experience]) -> float:
        """
        Update the policy based on collected experiences
        Returns the policy improvement metric
        """
        # Base implementation is a no-op
        # Subclasses should implement specific learning algorithms
        return 0.0
    
    def _load_policy(self) -> None:
        """Load policy from disk if it exists"""
        if self.model_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.parameters = data.get("parameters", {})
                    self.training_iterations = data.get("training_iterations", 0)
                    self.last_training_time = data.get("last_training_time")
                    self.cumulative_reward = data.get("cumulative_reward", 0)
                    self.version = data.get("version", 1)
                
                logger.info(f"Loaded policy for {self.agent_name} version {self.version}")
            except Exception as e:
                logger.error(f"Error loading policy for {self.agent_name}: {e}")
                self.parameters = {}
    
    def save_policy(self) -> None:
        """Save policy to disk"""
        try:
            with open(self.model_path, 'wb') as f:
                data = {
                    "parameters": self.parameters,
                    "training_iterations": self.training_iterations,
                    "last_training_time": datetime.now(),
                    "cumulative_reward": self.cumulative_reward,
                    "version": self.version
                }
                pickle.dump(data, f)
            
            logger.info(f"Saved policy for {self.agent_name} version {self.version}")
        except Exception as e:
            logger.error(f"Error saving policy for {self.agent_name}: {e}")


class RuleBasedPolicy(AgentPolicy):
    """
    Simple rule-based policy with parameter tuning via RL
    """
    def __init__(self, agent_name: str, model_dir: str = "./rl_models"):
        super().__init__(agent_name, model_dir)
        
        # Initialize default parameters if not loaded
        if not self.parameters:
            self.parameters = {
                # User interaction agent parameters
                "intent_confidence_threshold": 0.7,
                "conversation_context_window": 5,
                
                # Planning agent parameters
                "preference_weight": 0.7,
                "budget_weight": 0.8,
                "diversity_weight": 0.5,
                
                # Booking agent parameters
                "price_weight": 0.6,
                "rating_weight": 0.4,
                "booking_retry_limit": 3,
                
                # Monitoring agent parameters
                "price_alert_threshold": 0.1,  # 10% change
                "monitoring_frequency_hours": 12,
            }
    
    def select_action(self, state_features: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Select action based on rule-based policy with learned parameters"""
        action = "default_action"
        params = {}
        
        # Different actions for different agents
        if self.agent_name == "user_interaction_agent":
            # User interaction agent selects intent analysis or clarification
            if state_features.get("query_complexity", 0) > 0.7:
                action = "request_clarification"
                params = {"confidence_threshold": self.parameters["intent_confidence_threshold"]}
            else:
                action = "analyze_intent"
                params = {
                    "confidence_threshold": self.parameters["intent_confidence_threshold"],
                    "context_window": self.parameters["conversation_context_window"]
                }
                
        elif self.agent_name == "planning_agent":
            # Planning agent selects itinerary creation or modification
            if state_features.get("has_itinerary", False):
                action = "modify_itinerary"
            else:
                action = "create_itinerary"
            
            params = {
                "preference_weight": self.parameters["preference_weight"],
                "budget_weight": self.parameters["budget_weight"],
                "diversity_weight": self.parameters["diversity_weight"]
            }
                
        elif self.agent_name == "booking_agent":
            # Booking agent selects booking strategy
            action = "book_itinerary_items"
            params = {
                "price_weight": self.parameters["price_weight"],
                "rating_weight": self.parameters["rating_weight"],
                "retry_limit": self.parameters["booking_retry_limit"]
            }
                
        elif self.agent_name == "monitoring_agent":
            # Monitoring agent selects monitoring strategy
            action = "monitor_prices_and_availability"
            params = {
                "price_alert_threshold": self.parameters["price_alert_threshold"],
                "frequency_hours": self.parameters["monitoring_frequency_hours"]
            }
        
        return action, params
    
    def update_policy(self, experiences: List[Experience]) -> float:
        """Update policy parameters based on experiences"""
        if not experiences:
            return 0.0
        
        # Calculate average reward
        avg_reward = sum(exp.reward for exp in experiences) / len(experiences)
        
        # Simple parameter adjustment based on rewards
        if avg_reward > 0:
            # Positive reinforcement - adjust parameters slightly in the direction that worked
            for exp in experiences:
                if exp.reward > 0:
                    action_params = exp.action_params
                    for param_name, param_value in action_params.items():
                        if param_name in self.parameters:
                            # Move parameter slightly toward the value that worked well
                            current_value = self.parameters[param_name]
                            adjustment = 0.1 * exp.reward  # Scale adjustment by reward
                            self.parameters[param_name] = current_value + adjustment * (param_value - current_value)
        
        # Update training statistics
        self.training_iterations += 1
        self.last_training_time = datetime.now()
        self.cumulative_reward += avg_reward
        
        # Save updated policy
        self.save_policy()
        
        return avg_reward


class RLTrainer:
    """
    Manages the training of agent policies using collected experiences
    """
    def __init__(
        self,
        experience_buffer: ExperienceBuffer,
        policies: Dict[str, AgentPolicy],
        training_config: Dict[str, Any] = None
    ):
        self.experience_buffer = experience_buffer
        self.policies = policies
        
        # Default training configuration
        self.config = {
            "batch_size": 64,
            "training_frequency": 100,  # Train every N experiences
            "min_experiences": 500,  # Minimum experiences before training
            "save_frequency": 10,  # Save model every N training iterations
            "validation_split": 0.2,  # Portion of experiences used for validation
        }
        
        # Update with provided configuration
        if training_config:
            self.config.update(training_config)
        
        # Training metrics
        self.metrics = {agent_name: {
            "training_count": 0,
            "avg_reward": 0,
            "last_batch_reward": 0,
            "improvement_rate": 0
        } for agent_name in self.policies.keys()}
        
        logger.info(f"RL Trainer initialized with config: {self.config}")
    
    def train_agent(self, agent_name: str) -> Dict[str, float]:
        """Train a specific agent policy"""
        if agent_name not in self.policies:
            logger.warning(f"No policy defined for agent: {agent_name}")
            return {}
        
        # Check if we have enough experiences
        if self.experience_buffer.experience_counts.get(agent_name, 0) < self.config["min_experiences"]:
            logger.info(f"Not enough experiences for {agent_name}. Have {self.experience_buffer.experience_counts.get(agent_name, 0)}, need {self.config['min_experiences']}")
            return {}
        
        # Sample training batch
        batch = self.experience_buffer.sample_batch(agent_name, self.config["batch_size"])
        
        if not batch:
            return {}
        
        # Update policy
        policy = self.policies[agent_name]
        improvement = policy.update_policy(batch)
        
        # Update metrics
        self.metrics[agent_name]["training_count"] += 1
        self.metrics[agent_name]["last_batch_reward"] = sum(exp.reward for exp in batch) / len(batch)
        self.metrics[agent_name]["avg_reward"] = (self.metrics[agent_name]["avg_reward"] * 
                                               (self.metrics[agent_name]["training_count"] - 1) + 
                                               self.metrics[agent_name]["last_batch_reward"]) / self.metrics[agent_name]["training_count"]
        self.metrics[agent_name]["improvement_rate"] = improvement
        
        # Save policy periodically
        if self.metrics[agent_name]["training_count"] % self.config["save_frequency"] == 0:
            policy.save_policy()
        
        logger.info(f"Trained {agent_name} policy: reward={self.metrics[agent_name]['last_batch_reward']:.4f}, improvement={improvement:.4f}")
        
        return {
            "avg_reward": self.metrics[agent_name]["avg_reward"],
            "last_batch_reward": self.metrics[agent_name]["last_batch_reward"],
            "improvement_rate": improvement,
            "training_count": self.metrics[agent_name]["training_count"]
        }
    
    def train_all_agents(self) -> Dict[str, Dict[str, float]]:
        """Train all agent policies"""
        results = {}
        for agent_name in self.policies.keys():
            results[agent_name] = self.train_agent(agent_name)
        return results
    
    def should_train(self, agent_name: str, new_experiences: int) -> bool:
        """Determine if it's time to train based on new experiences"""
        # Skip if we don't have enough experiences
        if self.experience_buffer.experience_counts.get(agent_name, 0) < self.config["min_experiences"]:
            return False
        
        # Train if we've received enough new experiences
        policy = self.policies.get(agent_name)
        if policy:
            return (self.experience_buffer.experience_counts.get(agent_name, 0) - 
                   policy.training_iterations * self.config["batch_size"]) >= self.config["training_frequency"]
        
        return False


class RLController:
    """
    Main controller that integrates agents, experience collection, and training
    """
    def __init__(self, config_path: str = None):
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.experience_buffer = ExperienceBuffer(
            buffer_size=self.config.get("buffer_size", 100000),
            storage_path=self.config.get("storage_path", "./rl_data")
        )
        
        # Initialize policies for each agent
        self.policies = {
            "user_interaction_agent": RuleBasedPolicy("user_interaction_agent"),
            "planning_agent": RuleBasedPolicy("planning_agent"),
            "booking_agent": RuleBasedPolicy("booking_agent"),
            "monitoring_agent": RuleBasedPolicy("monitoring_agent")
        }
        
        # Initialize trainer
        self.trainer = RLTrainer(
            experience_buffer=self.experience_buffer,
            policies=self.policies,
            training_config=self.config.get("training", {})
        )
        
        # State tracking
        self.current_state = {}
        self.current_actions = {}
        
        logger.info("RLController initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        config = {
            "buffer_size": 100000,
            "storage_path": "./rl_data",
            "model_dir": "./rl_models",
            "logging_level": "INFO",
            "training": {
                "batch_size": 64,
                "training_frequency": 100,
                "min_experiences": 500,
                "save_frequency": 10,
            },
            "reward_weights": {
                "user_satisfaction": 1.0,
                "itinerary_quality": 0.8,
                "booking_efficiency": 0.7,
                "monitoring_effectiveness": 0.6
            }
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Update config with loaded values
                    for key, value in loaded_config.items():
                        if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                            # Merge nested dictionaries
                            config[key].update(value)
                        else:
                            # Replace top-level values
                            config[key] = value
            except Exception as e:
                logger.error(f"Error loading configuration from {config_path}: {e}")
        
        return config
    
    def select_agent_action(self, agent_name: str, state_features: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Select an action for a specific agent based on current state"""
        if agent_name not in self.policies:
            logger.warning(f"No policy for agent: {agent_name}")
            return "default_action", {}
        
        policy = self.policies[agent_name]
        action_name, action_params = policy.select_action(state_features)
        
        # Record state and action for later experience collection
        self.current_state[agent_name] = state_features
        self.current_actions[agent_name] = (action_name, action_params)
        
        return action_name, action_params
    
    def record_experience(
        self, 
        agent_name: str, 
        next_state_features: Dict[str, Any],
        reward_data: Dict[str, Any], 
        done: bool = False
    ) -> None:
        """Record an experience for an agent based on action results"""
        if agent_name not in self.current_state or agent_name not in self.current_actions:
            logger.warning(f"Missing state or action for agent: {agent_name}")
            return
        
        # Calculate reward based on agent type and provided data
        reward = self._calculate_reward(agent_name, reward_data)
        
        # Create experience
        state_repr = self.current_state[agent_name]
        action_name, action_params = self.current_actions[agent_name]
        
        experience = Experience(
            state_repr=state_repr,
            action_name=action_name,
            action_params=action_params,
            reward=reward,
            next_state_repr=next_state_features,
            done=done,
            metadata={"reward_data": reward_data}
        )
        
        # Add to buffer
        self.experience_buffer.add_experience(agent_name, experience)
        
        # Check if we should train
        new_experiences = self.experience_buffer.experience_counts.get(agent_name, 0)
        if self.trainer.should_train(agent_name, new_experiences):
            self.trainer.train_agent(agent_name)
        
        # Update current state
        self.current_state[agent_name] = next_state_features
    
    def _calculate_reward(self, agent_name: str, reward_data: Dict[str, Any]) -> float:
        """Calculate reward for an agent based on reward data"""
        reward = 0.0
        weights = self.config.get("reward_weights", {})
        
        if agent_name == "user_interaction_agent":
            # User interaction agent gets rewarded for user satisfaction
            user_satisfaction = RewardFunction.calculate_user_satisfaction(reward_data.get("user_feedback"))
            reward = user_satisfaction * weights.get("user_satisfaction", 1.0)
            
        elif agent_name == "planning_agent":
            # Planning agent gets rewarded for itinerary quality
            itinerary_quality = RewardFunction.calculate_itinerary_quality(
                reward_data.get("itinerary"), 
                reward_data.get("user_preferences")
            )
            reward = itinerary_quality * weights.get("itinerary_quality", 0.8)
            
        elif agent_name == "booking_agent":
            # Booking agent gets rewarded for booking efficiency
            booking_efficiency = RewardFunction.calculate_booking_efficiency(
                reward_data.get("bookings"),
                reward_data.get("itinerary")
            )
            reward = booking_efficiency * weights.get("booking_efficiency", 0.7)
            
        elif agent_name == "monitoring_agent":
            # Monitoring agent gets rewarded for monitoring effectiveness
            monitoring_effectiveness = RewardFunction.calculate_monitoring_effectiveness(
                reward_data.get("alerts", []),
                reward_data.get("resolved_issues", [])
            )
            reward = monitoring_effectiveness * weights.get("monitoring_effectiveness", 0.6)
        
        return reward
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get current training metrics for all agents"""
        return {
            "training_counts": {agent: metrics["training_count"] for agent, metrics in self.trainer.metrics.items()},
            "avg_rewards": {agent: metrics["avg_reward"] for agent, metrics in self.trainer.metrics.items()},
            "experience_counts": self.experience_buffer.experience_counts,
            "policy_versions": {agent: policy.version for agent, policy in self.policies.items()}
        }

# Usage in production with Travel Agent System
class RLEnhancedTravelSystem:
    """
    Enhanced travel agent system that integrates RL-based learning
    """
    def __init__(self, base_system, rl_config_path: str = None):
        """
        Initialize the RL-enhanced system
        
        Args:
            base_system: The base travel agent system to enhance
            rl_config_path: Path to RL configuration file
        """
        self.base_system = base_system
        self.rl_controller = RLController(rl_config_path)
        
        # Setup state extractors
        self.state_extractor = StateFeatureExtractor()
        
        # Experience collection tracking
        self.session_states = {}
        self.session_actions = {}
        self.session_feedback = {}
        
        logger.info("RL-enhanced travel system initialized")
    
    def process_user_request(self, user_id: str, user_input: str) -> Dict[str, Any]:
        """
        Process a user request with RL-enhanced agent selection
        
        Args:
            user_id: User identifier
            user_input: User's request message
            
        Returns:
            Response including itinerary, messages, and alerts
        """
        # Store initial state for all agents
        initial_team_memory = self.base_system.team_memory
        initial_states = {
            agent_name: self.state_extractor.extract_state_features(initial_team_memory)
            for agent_name in self.rl_controller.policies.keys()
        }
        self.session_states = initial_states
        
        # Get RL-optimized parameters for each agent
        agent_parameters = {}
        for agent_name in self.rl_controller.policies.keys():
            action_name, action_params = self.rl_controller.select_agent_action(
                agent_name, 
                self.session_states[agent_name]
            )
            agent_parameters[agent_name] = action_params
            self.session_actions[agent_name] = (action_name, action_params)
        
        # Apply optimized parameters to base system
        # Note: This requires base system to accept these parameters
        for agent_name, params in agent_parameters.items():
            if hasattr(self.base_system, f"configure_{agent_name}"):
                getattr(self.base_system, f"configure_{agent_name}")(params)
        
        # Process the request with enhanced parameters
        result = self.base_system.process_user_request(user_id, user_input)
        
        # Extract final states after processing
        final_team_memory = self.base_system.team_memory
        final_states = {
            agent_name: self.state_extractor.extract_state_features(final_team_memory)
            for agent_name in self.rl_controller.policies.keys()
        }
        
        # Prepare reward data for each agent
        reward_data = {
            "user_interaction_agent": {
                "user_feedback": self._extract_user_feedback(result)
            },
            "planning_agent": {
                "itinerary": final_team_memory.current_itinerary,
                "user_preferences": final_team_memory.user_profile
            },
            "booking_agent": {
                "bookings": final_team_memory.booking_history,
                "itinerary": final_team_memory.current_itinerary
            },
            "monitoring_agent": {
                "alerts": final_team_memory.monitoring_alerts,
                "resolved_issues": [a for a in final_team_memory.monitoring_alerts if getattr(a, "resolved", False)]
            }
        }
        
        # Record experiences
        for agent_name in self.rl_controller.policies.keys():
            done = agent_name != "monitoring_agent"  # Monitoring agent continues across sessions
            self.rl_controller.record_experience(
                agent_name,
                final_states[agent_name],
                reward_data[agent_name],
                done
            )
        
        # Attach RL insights to result
        result["rl_insights"] = {
            "training_metrics": self.rl_controller.get_training_metrics(),
            "agent_parameters": agent_parameters
        }
        
        return result
    
    def _extract_user_feedback(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract implicit user feedback from results"""
        feedback = {
            "itinerary_accepted": result.get("itinerary") is not None,
            "booking_successful": len(result.get("messages", [])) > 0,
            "errors": len([m for m in result.get("messages", []) 
                         if m.get("message_type") == "error"]),
            "satisfaction_score": 0.0
        }
        
        # Implicit satisfaction from complete itinerary
        if feedback["itinerary_accepted"]:
            feedback["satisfaction_score"] += 0.5
        
        # Implicit satisfaction from successful bookings
        booking_messages = [m for m in result.get("messages", []) 
                          if m.get("message_type") == "booking_confirmation"]
        if booking_messages:
            feedback["satisfaction_score"] += 0.3
            feedback["booking_successful"] = True
        
        # Implicit dissatisfaction from alerts
        if result.get("alerts"):
            feedback["satisfaction_score"] -= 0.1 * min(len(result.get("alerts")), 5)
        
        return feedback
    
    def provide_explicit_feedback(self, user_id: str, feedback: Dict[str, Any]) -> None:
        """
        Allow users to provide explicit feedback about their experience
        
        Args:
            user_id: User identifier
            feedback: Dictionary with feedback data including satisfaction score,
                     itinerary rating, etc.
        """
        if user_id not in self.session_feedback:
            self.session_feedback[user_id] = []
        
        self.session_feedback[user_id].append(feedback)
        
        # Update reward calculations with explicit feedback
        if feedback.get("satisfaction_score") is not None:
            # Recalculate user interaction agent reward with explicit feedback
            self.rl_controller.record_experience(
                "user_interaction_agent",
                self.session_states.get("user_interaction_agent", {}),
                {"user_feedback": feedback},
                True
            )
        
        logger.info(f"Received explicit feedback from user {user_id}: {feedback}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics including RL training status"""
        metrics = {
            "rl_training_metrics": self.rl_controller.get_training_metrics(),
            "base_system_metrics": getattr(self.base_system, "get_performance_metrics", lambda: {})()
        }
        return metrics
    
    def save_models(self) -> None:
        """Save all RL models"""
        for agent_name, policy in self.rl_controller.policies.items():
            policy.save_policy()
        logger.info("All RL models saved")


# Create configuration file
def create_default_config() -> None:
    """Create a default configuration file if one doesn't exist"""
    config_path = Path("./rl_config.json")
    
    if not config_path.exists():
        default_config = {
            "buffer_size": 100000,
            "storage_path": "./rl_data",
            "model_dir": "./rl_models",
            "logging_level": "INFO",
            "training": {
                "batch_size": 64,
                "training_frequency": 100,
                "min_experiences": 500,
                "save_frequency": 10,
            },
            "reward_weights": {
                "user_satisfaction": 1.0,
                "itinerary_quality": 0.8,
                "booking_efficiency": 0.7,
                "monitoring_effectiveness": 0.6
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        logger.info(f"Created default RL configuration at {config_path}")


# Example usage
if __name__ == "__main__":
    # Create default configuration if needed
    create_default_config()
    
    # Initialize components for testing
    controller = RLController("./rl_config.json")
    
    # Print training metrics
    metrics = controller.get_training_metrics()
    print(f"Training metrics: {metrics}")
    
    print("RL framework initialized successfully")
    print("Import this module in your production system to enable RL-based learning")