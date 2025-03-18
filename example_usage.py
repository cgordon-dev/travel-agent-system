"""
Example usage of the Travel & Event Planning System
Demonstrates how the system processes a user request through the agent workflow
"""

import json
import uuid
from datetime import datetime, timedelta

from pydantic_models import (
    UserPreference, 
    Location, 
    PriceRange, 
    AccommodationType, 
    ActivityType, 
    TransportationType,
    TeamMemory
)
from travel_agent_architecture import TravelPlanningSystem

def create_sample_user_profile():
    """Create a sample user profile for demonstration"""
    
    # Create user preferences
    user_prefs = UserPreference(
        traveler_profile={
            "name": "John Doe",
            "email": "john.doe@example.com",
            "age": 35,
            "family_size": 4,
            "previous_destinations": ["London", "Rome", "Tokyo"]
        },
        accommodation_types=[
            AccommodationType.HOTEL,
            AccommodationType.RESORT
        ],
        transportation_types=[
            TransportationType.FLIGHT,
            TransportationType.CAR_RENTAL
        ],
        activity_types=[
            ActivityType.CULTURAL,
            ActivityType.ADVENTURE, 
            ActivityType.CULINARY
        ],
        budget=PriceRange(
            min_price=1000,
            max_price=5000,
            currency="USD"
        ),
        accessibility_requirements=["wheelchair accessible"],
        dietary_restrictions=["vegetarian options"]
    )
    
    # Add some learned preferences
    user_prefs.update_preference("prefers_window_seat", True, 0.9)
    user_prefs.update_preference("morning_activities", True, 0.8)
    user_prefs.update_preference("preferred_airline", "SkyWings", 0.7)
    
    return user_prefs

def demo_paris_trip():
    """Demonstrate planning a family trip to Paris"""
    
    # Create the system
    system = TravelPlanningSystem()
    
    # Set up the user profile
    user_profile = create_sample_user_profile()
    system.team_memory.user_profile = user_profile
    
    # Process a user request
    print("\n=== User Request ===")
    user_input = "I want to plan a trip to Paris for next week with my family. We're interested in cultural activities and prefer to stay in a hotel near the Eiffel Tower."
    print(user_input)
    
    # Generate a unique user ID
    user_id = f"user_{uuid.uuid4().hex[:8]}"
    
    # Process the request
    result = system.process_user_request(user_id, user_input)
    
    # Display the result
    print("\n=== System Response ===")
    print(json.dumps({
        "itinerary_id": result.get("itinerary", {}).get("itinerary_id", "N/A"),
        "title": result.get("itinerary", {}).get("title", "N/A"),
        "start_date": result.get("itinerary", {}).get("start_date", "N/A"),
        "end_date": result.get("itinerary", {}).get("end_date", "N/A"),
        "total_cost": result.get("itinerary", {}).get("total_cost", "N/A"),
        "message_count": len(result.get("messages", [])),
        "alert_count": len(result.get("alerts", []))
    }, indent=2))
    
    # Display the first few activities
    if result.get("itinerary") and result.get("itinerary", {}).get("activities"):
        print("\n=== Sample Activities ===")
        for i, activity in enumerate(result.get("itinerary").get("activities")[:3], 1):
            print(f"{i}. {activity.get('description')} - {activity.get('activity_type')}")
    
    # Display any alerts
    if result.get("alerts"):
        print("\n=== Alerts ===")
        for alert in result.get("alerts"):
            print(f"- {alert.get('alert_type')}: {alert.get('description')}")
    
    return result

def demo_booking_changes():
    """Demonstrate the system handling booking changes and monitoring"""
    
    # Create the system with a pre-existing itinerary
    system = TravelPlanningSystem()
    
    # Set up the user profile
    user_profile = create_sample_user_profile()
    system.team_memory.user_profile = user_profile
    
    # Initial setup with a pre-existing itinerary
    result = system.process_user_request(
        "user123", 
        "I already have a trip to Rome booked next month, but I'd like to change my hotel to something closer to the Colosseum"
    )
    
    # Display the result
    print("\n=== Booking Change Request ===")
    print(json.dumps({
        "booking_changes": [msg.get("content", {}).get("summary") for msg in result.get("messages", [])
                           if msg.get("message_type") == "booking_confirmation"],
        "alerts": [alert.get("description") for alert in result.get("alerts", [])]
    }, indent=2))
    
    return result

def demo_price_monitoring():
    """Demonstrate the price monitoring functionality"""
    
    # Create the system
    system = TravelPlanningSystem()
    
    # Set up the user profile
    user_profile = create_sample_user_profile()
    system.team_memory.user_profile = user_profile
    
    # Process a monitoring request
    result = system.process_user_request(
        "user123", 
        "Can you monitor my upcoming trip to Tokyo for any price drops or cancellation issues?"
    )
    
    # Display the result
    print("\n=== Monitoring Request ===")
    print(json.dumps({
        "monitoring_setup": [msg.get("content", {}).get("summary") for msg in result.get("messages", [])
                           if msg.get("message_type") == "status_update"],
        "alerts": [f"{alert.get('alert_type')}: {alert.get('description')}" for alert in result.get("alerts", [])]
    }, indent=2))
    
    return result

if __name__ == "__main__":
    print("=== Travel & Event Planning System Demo ===")
    print("This demonstrates how the multi-agent system processes travel requests")
    
    # Note: In a real implementation, these functions would interact with actual APIs
    # and perform real processing. The current implementation returns simulated data
    # for demonstration purposes.
    
    # Run the demos
    paris_trip = demo_paris_trip()
    booking_changes = demo_booking_changes()
    price_monitoring = demo_price_monitoring()
    
    print("\n=== Demo Complete ===")
    print("In a full implementation, these examples would trigger the complete agent workflow")
    print("with real API calls to travel services and genuine itinerary creation.")