"""
Example demonstrating how to use multiple LLM providers with the Travel Agent System
Shows how to switch between OpenAI and Anthropic models for different agents
"""

import os
import json
from travel_agent_architecture import TravelPlanningSystem
from llm_config import LLMProvider, ModelType
from datetime import datetime, timedelta

def create_env_file():
    """Create a .env file template if one doesn't exist"""
    if not os.path.exists(".env") and os.path.exists(".env.example"):
        print("Creating .env file from .env.example...")
        with open(".env.example", "r") as example:
            content = example.read()
        
        with open(".env", "w") as env_file:
            env_file.write(content)
        
        print("\033[93mWARNING: Please edit the .env file to add your API keys\033[0m")
        print("You need at least one of OpenAI or Anthropic API keys to continue.")

def show_model_usage(system):
    """Show which LLM models are being used by each agent"""
    config = system.get_current_llm_configuration()
    
    print("\n=== Current LLM Configuration ===")
    print(f"Available Providers: {', '.join(config['available_providers'])}")
    print("\nAgent Configuration:")
    
    for agent, provider in config["agent_providers"].items():
        model_type = config["agent_model_types"].get(agent, "balanced")
        print(f"  - {agent}: Provider: {provider}, Model Type: {model_type}")

def test_with_single_provider(provider):
    """Test the system using a single LLM provider for all agents"""
    print(f"\n=== Testing with {provider.value} as the provider for all agents ===")
    
    # Create a configuration override that uses the same provider for all agents
    config_override = {
        "agent_providers": {
            "user_interaction_agent": provider,
            "planning_agent": provider,
            "booking_agent": provider,
            "monitoring_agent": provider
        }
    }
    
    # Initialize the system with our configuration
    system = TravelPlanningSystem(llm_config_override=config_override)
    show_model_usage(system)
    
    # Process a simple request
    user_input = "I want to plan a weekend trip to New York City next month."
    user_id = "test_user_123"
    
    print(f"\nSending request: \"{user_input}\"")
    result = system.process_user_request(user_id, user_input)
    
    # Display a summary of the results
    print("\nRequest processed successfully!")
    
    if result.get("itinerary"):
        print(f"Itinerary created: {result['itinerary'].title}")
        print(f"Dates: {result['itinerary'].start_date.strftime('%Y-%m-%d')} to "
              f"{result['itinerary'].end_date.strftime('%Y-%m-%d')}")
    else:
        print("No itinerary was created")
    
    messages = result.get("messages", [])
    print(f"Messages: {len(messages)}")
    
    return system

def test_with_mixed_providers():
    """Test the system using different LLM providers for different agents"""
    print("\n=== Testing with mixed providers ===")
    print("Using OpenAI for planning and booking, Anthropic for user interaction and monitoring")
    
    # Create a configuration that uses different providers
    config_override = {
        "agent_providers": {
            "user_interaction_agent": LLMProvider.ANTHROPIC,
            "planning_agent": LLMProvider.OPENAI,
            "booking_agent": LLMProvider.OPENAI,
            "monitoring_agent": LLMProvider.ANTHROPIC
        },
        "agent_model_types": {
            "user_interaction_agent": ModelType.POWERFUL,  # Use most powerful model for user interaction
            "planning_agent": ModelType.POWERFUL,
            "booking_agent": ModelType.BALANCED,  # Use balanced model for booking (cost efficiency)
            "monitoring_agent": ModelType.BALANCED
        }
    }
    
    # Initialize the system with our configuration
    system = TravelPlanningSystem(llm_config_override=config_override)
    show_model_usage(system)
    
    # Process a slightly more complex request
    user_input = ("I need to plan a business trip to San Francisco for a conference "
                 "next week. I'll need a hotel near the Moscone Center and would "
                 "prefer to fly direct.")
    user_id = "business_user_456"
    
    print(f"\nSending request: \"{user_input}\"")
    result = system.process_user_request(user_id, user_input)
    
    # Display a summary of the results
    print("\nRequest processed successfully!")
    
    if result.get("itinerary"):
        print(f"Itinerary created: {result['itinerary'].title}")
        print(f"Dates: {result['itinerary'].start_date.strftime('%Y-%m-%d')} to "
              f"{result['itinerary'].end_date.strftime('%Y-%m-%d')}")
        print(f"Total cost: ${result['itinerary'].total_cost:.2f}")
        
        # Show accommodations
        if result['itinerary'].accommodations:
            print("\nSelected accommodation:")
            acc = result['itinerary'].accommodations[0]
            print(f"  - {acc.description} ({acc.accommodation_type})")
            
        # Show transportation
        if result['itinerary'].transportations:
            print("\nSelected transportation:")
            for i, transport in enumerate(result['itinerary'].transportations[:2]):
                print(f"  {i+1}. {transport.description} ({transport.transportation_type})")
    else:
        print("No itinerary was created")
    
    return system

def test_dynamic_switching(system):
    """Test dynamically switching a provider for an agent"""
    print("\n=== Testing dynamic provider switching ===")
    
    # Show current configuration
    show_model_usage(system)
    
    # Switch the planning agent to use a different provider
    current_provider = system.get_current_llm_configuration()["agent_providers"]["planning_agent"]
    
    # Determine which provider to switch to
    if current_provider == "openai":
        new_provider = LLMProvider.ANTHROPIC
    else:
        new_provider = LLMProvider.OPENAI
    
    print(f"\nSwitching planning_agent from {current_provider} to {new_provider.value}")
    system.configure_agent_provider("planning_agent", new_provider)
    
    # Show updated configuration
    show_model_usage(system)
    
    # Process another request to test the new configuration
    user_input = ("I'm planning a family vacation to Disney World in Orlando "
                 "for the summer. We need a family-friendly hotel and activities "
                 "for kids aged 5-10.")
    user_id = "family_user_789"
    
    print(f"\nSending request with new configuration: \"{user_input}\"")
    result = system.process_user_request(user_id, user_input)
    
    # Display a summary of the results
    print("\nRequest processed successfully!")
    
    if result.get("itinerary"):
        print(f"Itinerary created: {result['itinerary'].title}")
        print(f"Dates: {result['itinerary'].start_date.strftime('%Y-%m-%d')} to "
              f"{result['itinerary'].end_date.strftime('%Y-%m-%d')}")
        
        # Show activities
        if result['itinerary'].activities:
            print("\nSelected activities:")
            for i, activity in enumerate(result['itinerary'].activities[:3]):
                print(f"  {i+1}. {activity.description} ({activity.activity_type})")
    else:
        print("No itinerary was created")

if __name__ == "__main__":
    print("=== Travel Agent System with Multiple LLM Providers ===")
    print("This example demonstrates switching between OpenAI and Anthropic models")
    
    # Create .env file if needed
    create_env_file()
    
    try:
        # Test with OpenAI
        openai_system = test_with_single_provider(LLMProvider.OPENAI)
        
        # Test with mixed providers
        mixed_system = test_with_mixed_providers()
        
        # Test dynamically switching providers
        test_dynamic_switching(mixed_system)
        
        print("\n=== Example Complete ===")
        print("Successfully demonstrated the use of multiple LLM providers")
        
    except Exception as e:
        print(f"\n\033[91mError: {str(e)}\033[0m")
        print("\nMake sure you've added your API keys to the .env file.")
        print("You need at least one of OpenAI or Anthropic API keys to run this example.")