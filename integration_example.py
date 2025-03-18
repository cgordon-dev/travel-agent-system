"""
Integration Example for the Travel Agent System with Reinforcement Learning
Demonstrates how to apply RL to the base system for production use
"""

import os
import logging
import json
from datetime import datetime
from typing import Dict, Any, List

from travel_agent_architecture import TravelPlanningSystem
from reinforcement_learning import RLEnhancedTravelSystem
from pydantic_models import UserPreference, TravelItinerary

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("travel_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("integration")

class ProductionTravelSystem:
    """
    Production-ready travel planning system with RL enhancement
    Demonstrates the integration of the base system with RL capabilities
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the production system
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Set up base system
        self.base_system = TravelPlanningSystem()
        
        # Initialize the RL enhancement layer
        self.rl_system = RLEnhancedTravelSystem(
            base_system=self.base_system,
            rl_config_path=self.config.get("rl_config_path")
        )
        
        # Setup data storage
        self.data_dir = os.path.join(os.getcwd(), self.config.get("data_dir", "data"))
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize user database simulation (in production, use a real database)
        self.user_db = self._load_user_database()
        
        logger.info("Production travel system initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            "data_dir": "data",
            "rl_config_path": "./rl_config.json",
            "feedback_prompt_frequency": 3,  # Ask for feedback every N requests
            "save_frequency": 10,  # Save models every N requests
            "default_timeout": 60,  # Default timeout in seconds
            "logging_level": "INFO"
        }
        
        if not config_path:
            return default_config
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return default_config
    
    def _load_user_database(self) -> Dict[str, Dict[str, Any]]:
        """Load user database from disk (simulated)"""
        db_path = os.path.join(self.data_dir, "user_db.json")
        if os.path.exists(db_path):
            try:
                with open(db_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading user database: {e}")
                return {}
        return {}
    
    def _save_user_database(self) -> None:
        """Save user database to disk (simulated)"""
        db_path = os.path.join(self.data_dir, "user_db.json")
        try:
            with open(db_path, 'w') as f:
                json.dump(self.user_db, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving user database: {e}")
    
    def _get_user_profile(self, user_id: str) -> UserPreference:
        """Get or create user profile"""
        if user_id not in self.user_db:
            # Initialize new user profile
            self.user_db[user_id] = {
                "created_at": datetime.now().isoformat(),
                "request_count": 0,
                "last_seen": datetime.now().isoformat(),
                "feedback_count": 0,
                "profile": {}
            }
        
        # Update user stats
        self.user_db[user_id]["request_count"] += 1
        self.user_db[user_id]["last_seen"] = datetime.now().isoformat()
        
        # Create default user preference model
        user_prefs = UserPreference()
        
        # Fill in from stored profile if available
        stored_profile = self.user_db[user_id].get("profile", {})
        for key, value in stored_profile.items():
            if hasattr(user_prefs, key):
                setattr(user_prefs, key, value)
        
        return user_prefs
    
    def _update_user_profile(self, user_id: str, preferences: UserPreference) -> None:
        """Update user profile in database"""
        if user_id not in self.user_db:
            return
        
        # Convert Pydantic model to dict and store
        self.user_db[user_id]["profile"] = preferences.dict()
        self._save_user_database()
    
    def process_request(self, user_id: str, user_input: str) -> Dict[str, Any]:
        """
        Process a user request through the RL-enhanced system
        
        Args:
            user_id: User identifier
            user_input: User's request message
            
        Returns:
            Response including itinerary, messages, alerts, and system information
        """
        # Get user profile
        user_profile = self._get_user_profile(user_id)
        
        # Set user profile in base system
        self.base_system.team_memory.user_profile = user_profile
        
        # Process through RL-enhanced system
        start_time = datetime.now()
        result = self.rl_system.process_user_request(user_id, user_input)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Update user profile with new preferences if they were learned
        if self.base_system.team_memory.user_profile:
            updated_profile = self.base_system.team_memory.user_profile
            self._update_user_profile(user_id, updated_profile)
        
        # Check if we should ask for feedback
        should_request_feedback = (
            self.user_db[user_id]["request_count"] % 
            self.config["feedback_prompt_frequency"] == 0
        )
        
        # Add system information
        result["system_info"] = {
            "processing_time": processing_time,
            "request_number": self.user_db[user_id]["request_count"],
            "request_feedback": should_request_feedback,
            "agent_performance": self.rl_system.get_performance_metrics()
        }
        
        # Periodically save model state
        if sum(user["request_count"] for user in self.user_db.values()) % self.config["save_frequency"] == 0:
            self.rl_system.save_models()
        
        return result
    
    def save_feedback(self, user_id: str, feedback: Dict[str, Any]) -> Dict[str, str]:
        """
        Save user feedback and use it to improve the system
        
        Args:
            user_id: User identifier
            feedback: Dictionary with feedback data
            
        Returns:
            Status message
        """
        if user_id not in self.user_db:
            return {"status": "error", "message": "User not found"}
        
        # Update user feedback stats
        self.user_db[user_id]["feedback_count"] += 1
        
        # Store feedback
        if "feedback_history" not in self.user_db[user_id]:
            self.user_db[user_id]["feedback_history"] = []
        
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "feedback": feedback
        }
        self.user_db[user_id]["feedback_history"].append(feedback_entry)
        
        # Save to database
        self._save_user_database()
        
        # Send to RL system for learning
        self.rl_system.provide_explicit_feedback(user_id, feedback)
        
        return {"status": "success", "message": "Feedback recorded"}
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system statistics for monitoring and analytics
        
        Returns:
            System statistics
        """
        # Count active users (seen in the last 30 days)
        thirty_days_ago = datetime.now().timestamp() - (30 * 24 * 60 * 60)
        active_users = sum(
            1 for user in self.user_db.values() 
            if datetime.fromisoformat(user["last_seen"]).timestamp() > thirty_days_ago
        )
        
        # Get RL metrics
        rl_metrics = self.rl_system.get_performance_metrics()
        
        stats = {
            "total_users": len(self.user_db),
            "active_users": active_users,
            "total_requests": sum(user["request_count"] for user in self.user_db.values()),
            "total_feedback": sum(user.get("feedback_count", 0) for user in self.user_db.values()),
            "average_requests_per_user": sum(user["request_count"] for user in self.user_db.values()) / max(1, len(self.user_db)),
            "rl_training_metrics": rl_metrics
        }
        
        return stats


class ProductionSystemServer:
    """
    Example of a server wrapper for the production system
    In a real deployment, this would be a web service using Flask, FastAPI, etc.
    """
    
    def __init__(self, config_path: str = None):
        self.travel_system = ProductionTravelSystem(config_path)
        logger.info("Production server initialized")
    
    def start(self, port: int = 8000) -> None:
        """
        Start the server (simulated)
        
        Args:
            port: Port to listen on
        """
        logger.info(f"Server would start on port {port}")
        logger.info("This is a simulation - in production, implement with Flask/FastAPI")
    
    def handle_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an API request (simulated)
        
        Args:
            request_data: Request data
            
        Returns:
            Response data
        """
        request_type = request_data.get("type")
        user_id = request_data.get("user_id")
        
        if not user_id:
            return {"error": "User ID is required"}
        
        if request_type == "travel_request":
            user_input = request_data.get("user_input")
            if not user_input:
                return {"error": "User input is required"}
            
            return self.travel_system.process_request(user_id, user_input)
        
        elif request_type == "feedback":
            feedback = request_data.get("feedback")
            if not feedback:
                return {"error": "Feedback data is required"}
            
            return self.travel_system.save_feedback(user_id, feedback)
        
        elif request_type == "system_stats":
            # Only allow admins to access system stats
            if request_data.get("is_admin"):
                return self.travel_system.get_system_stats()
            else:
                return {"error": "Unauthorized"}
        
        else:
            return {"error": "Unknown request type"}


# Example simulation of production usage
def simulate_production():
    """Simulate a production deployment"""
    
    # Create server
    server = ProductionSystemServer()
    
    # Simulate handling requests
    print("\n=== Simulating User 1 - First Trip Planning ===")
    response = server.handle_request({
        "type": "travel_request",
        "user_id": "user1",
        "user_input": "I want to plan a family trip to Paris next month for 5 days. We're interested in museums and good food."
    })
    print(f"Response status: {response.get('system_info', {}).get('processing_time', 0):.2f} seconds")
    
    # Print itinerary details if available
    if response.get("itinerary"):
        print(f"Itinerary created: {response['itinerary'].get('title', 'N/A')}")
        print(f"Total cost: {response['itinerary'].get('total_cost', 'N/A')}")
    
    # Simulate feedback
    print("\n=== Simulating User 1 - Providing Feedback ===")
    feedback_response = server.handle_request({
        "type": "feedback",
        "user_id": "user1",
        "feedback": {
            "satisfaction_score": 4.5,
            "comments": "Great suggestions, but I'd like more family-friendly activities.",
            "itinerary_rating": 4,
            "booking_experience": 5,
            "suggestions": "Add more information about accessibility"
        }
    })
    print(f"Feedback status: {feedback_response.get('status', 'error')}")
    
    # Simulate another user
    print("\n=== Simulating User 2 - Business Trip ===")
    response2 = server.handle_request({
        "type": "travel_request",
        "user_id": "user2",
        "user_input": "I need to plan a business trip to New York City next week. I'll need a hotel near the financial district and transportation to JFK airport."
    })
    print(f"Response status: {response2.get('system_info', {}).get('processing_time', 0):.2f} seconds")
    
    # Get system stats (admin only)
    print("\n=== Simulating Admin - System Stats ===")
    stats = server.handle_request({
        "type": "system_stats",
        "user_id": "admin",
        "is_admin": True
    })
    print("System Statistics:")
    print(f"- Total users: {stats.get('total_users', 0)}")
    print(f"- Active users: {stats.get('active_users', 0)}")
    print(f"- Total requests: {stats.get('total_requests', 0)}")
    print(f"- Total feedback submissions: {stats.get('total_feedback', 0)}")
    
    # Show RL training metrics
    rl_metrics = stats.get("rl_training_metrics", {}).get("rl_training_metrics", {})
    print("\nRL Training Metrics:")
    for agent, metrics in rl_metrics.get("training_counts", {}).items():
        print(f"- {agent}: {metrics} training iterations")
    
    print("\nProduction simulation complete!")


if __name__ == "__main__":
    print("=== Travel Agent System - Production Simulation ===")
    print("This example demonstrates the integration of the RL framework")
    print("with the travel agent system in a production environment")
    
    # Run simulation
    simulate_production()