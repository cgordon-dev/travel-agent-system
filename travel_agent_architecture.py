"""
Multi-Agent Travel & Event Planning System Architecture
Built with LangGraph and PydanticAI
"""

from typing import List, Dict, Any, Optional, Union, Literal
from datetime import datetime, timedelta
import json
import os
from enum import Enum
from pydantic import BaseModel, Field, validator, ConfigDict
from langgraph.graph import StateGraph, END
from pydanticai import Agent, System, Tool, model

# =============================================
# Core Data Models with PydanticAI Integration
# =============================================

class Location(BaseModel):
    city: str
    country: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class PriceRange(BaseModel):
    min_price: float
    max_price: float
    currency: str = "USD"

class AccommodationType(str, Enum):
    HOTEL = "hotel"
    APARTMENT = "apartment"
    HOSTEL = "hostel"
    RESORT = "resort"
    HOUSE = "house"

class TransportationType(str, Enum):
    FLIGHT = "flight"
    TRAIN = "train"
    BUS = "bus"
    CAR_RENTAL = "car_rental"
    FERRY = "ferry"

class ActivityType(str, Enum):
    SIGHTSEEING = "sightseeing"
    CULTURAL = "cultural"
    ADVENTURE = "adventure"
    CULINARY = "culinary"
    RELAXATION = "relaxation"
    ENTERTAINMENT = "entertainment"

class UserPreference(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    
    traveler_profile: Dict[str, Any] = Field(
        default_factory=dict,
        description="Profile containing user preferences, travel history, and behavioral patterns"
    )
    accommodation_types: List[AccommodationType] = Field(
        default_factory=list,
        description="Preferred types of accommodation"
    )
    transportation_types: List[TransportationType] = Field(
        default_factory=list,
        description="Preferred types of transportation"
    )
    activity_types: List[ActivityType] = Field(
        default_factory=list,
        description="Preferred types of activities"
    )
    budget: Optional[PriceRange] = None
    accessibility_requirements: List[str] = Field(default_factory=list)
    dietary_restrictions: List[str] = Field(default_factory=list)
    
    # Dynamic preference learning
    learned_preferences: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dynamically updated preferences based on user feedback and interactions"
    )
    
    # Schema evolution mechanism
    @classmethod
    def update_schema(cls, new_fields: Dict[str, Any]):
        """Dynamically extend the schema with new fields as agents learn new preference types"""
        for field_name, field_info in new_fields.items():
            if field_name not in cls.__annotations__:
                # Add new field to the model
                cls.__annotations__[field_name] = field_info["type"]
                # Add default value
                setattr(cls, field_name, field_info.get("default", None))

class TravelItineraryItem(BaseModel):
    day: int
    date: datetime
    location: Location
    description: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    booking_reference: Optional[str] = None
    booking_status: Optional[str] = None
    price: Optional[float] = None
    currency: str = "USD"
    notes: Optional[str] = None

class Accommodation(TravelItineraryItem):
    accommodation_type: AccommodationType
    check_in_time: datetime
    check_out_time: datetime
    amenities: List[str] = Field(default_factory=list)
    room_type: Optional[str] = None
    
class Transportation(TravelItineraryItem):
    transportation_type: TransportationType
    departure_location: Location
    arrival_location: Location
    operator: Optional[str] = None
    seat_details: Optional[str] = None
    
class Activity(TravelItineraryItem):
    activity_type: ActivityType
    duration: timedelta
    included_items: List[str] = Field(default_factory=list)
    meeting_point: Optional[str] = None

class TravelItinerary(BaseModel):
    itinerary_id: str
    user_id: str
    title: str
    start_date: datetime
    end_date: datetime
    accommodations: List[Accommodation] = Field(default_factory=list)
    transportations: List[Transportation] = Field(default_factory=list)
    activities: List[Activity] = Field(default_factory=list)
    total_cost: float = 0
    currency: str = "USD"
    status: str = "draft"
    
    def add_item(self, item: Union[Accommodation, Transportation, Activity]):
        """Add an item to the appropriate list based on its type"""
        if isinstance(item, Accommodation):
            self.accommodations.append(item)
        elif isinstance(item, Transportation):
            self.transportations.append(item)
        elif isinstance(item, Activity):
            self.activities.append(item)
        else:
            raise ValueError(f"Unknown item type: {type(item)}")
        
        # Update total cost
        if item.price:
            self.total_cost += item.price

class BookingConfirmation(BaseModel):
    booking_id: str
    service_type: str
    provider: str
    booking_status: str
    confirmation_code: str
    booking_date: datetime
    price: float
    currency: str = "USD"
    cancellation_policy: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)

class MonitoringAlert(BaseModel):
    alert_id: str
    timestamp: datetime
    alert_type: str
    severity: str
    affected_booking_id: Optional[str] = None
    description: str
    recommended_action: Optional[str] = None
    price_change: Optional[float] = None

class AgentMessage(BaseModel):
    message_id: str
    sender: str
    recipient: str
    timestamp: datetime
    message_type: str
    content: Dict[str, Any]
    requires_response: bool = False

class TeamMemory(BaseModel):
    """Shared memory for all agents to access common knowledge and context"""
    user_profile: UserPreference
    current_itinerary: Optional[TravelItinerary] = None
    booking_history: List[BookingConfirmation] = Field(default_factory=list)
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    monitoring_alerts: List[MonitoringAlert] = Field(default_factory=list)
    shared_context: Dict[str, Any] = Field(default_factory=dict)
    
    def update_memory(self, key: str, value: Any):
        """Update a specific section of the team memory"""
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self.shared_context[key] = value

# =============================================
# LangGraph Agent State Definitions
# =============================================

class GraphState(BaseModel):
    """State that is passed between nodes in the LangGraph workflow"""
    user_input: str = ""
    user_id: str = ""
    travel_itinerary: Optional[TravelItinerary] = None
    team_memory: TeamMemory = Field(default_factory=TeamMemory)
    current_agent: str = "user_interaction_agent"
    errors: List[str] = Field(default_factory=list)
    messages: List[AgentMessage] = Field(default_factory=list)
    waiting_for_user_input: bool = False
    
    # For conditional routing
    planning_needed: bool = False
    booking_needed: bool = False
    monitoring_needed: bool = False
    task_complete: bool = False

# =============================================
# Agent Definitions
# =============================================

class UserInteractionAgent(Agent):
    """Agent that handles direct user interactions, interprets requests, and manages the conversation flow"""
    
    def __call__(self, state: GraphState) -> GraphState:
        """Process user input and determine next steps"""
        user_input = state.user_input
        
        # Update conversation history
        state.team_memory.conversation_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now()
        })
        
        # Determine intent and required actions
        intent_analysis = self.analyze_intent(user_input)
        
        # Update state based on intent
        state.planning_needed = intent_analysis.get("planning_needed", False)
        state.booking_needed = intent_analysis.get("booking_needed", False)
        state.monitoring_needed = intent_analysis.get("monitoring_needed", False)
        
        # Create agent message
        message = AgentMessage(
            message_id=f"msg_{len(state.messages) + 1}",
            sender="user_interaction_agent",
            recipient="system",
            timestamp=datetime.now(),
            message_type="intent_analysis",
            content=intent_analysis,
            requires_response=False
        )
        state.messages.append(message)
        
        return state
    
    @model
    def analyze_intent(self, user_input: str) -> Dict[str, Any]:
        """Analyze user input to determine intent and required actions"""
        pass  # Implemented by PydanticAI

class PlanningAgent(Agent):
    """Agent responsible for creating and modifying travel itineraries"""
    
    def __call__(self, state: GraphState) -> GraphState:
        """Create or update a travel itinerary based on user preferences"""
        # Check if we need to create a new itinerary or update existing one
        if not state.travel_itinerary:
            # Create new itinerary
            itinerary = self.create_new_itinerary(
                user_id=state.user_id, 
                preferences=state.team_memory.user_profile
            )
            state.travel_itinerary = itinerary
        else:
            # Update existing itinerary
            updated_itinerary = self.update_itinerary(
                itinerary=state.travel_itinerary, 
                user_input=state.user_input,
                preferences=state.team_memory.user_profile
            )
            state.travel_itinerary = updated_itinerary
        
        # Update team memory
        state.team_memory.current_itinerary = state.travel_itinerary
        
        # Create agent message
        message = AgentMessage(
            message_id=f"msg_{len(state.messages) + 1}",
            sender="planning_agent",
            recipient="user_interaction_agent",
            timestamp=datetime.now(),
            message_type="itinerary_update",
            content={"itinerary": state.travel_itinerary},
            requires_response=True
        )
        state.messages.append(message)
        
        # Set booking needed
        state.booking_needed = True
        
        return state
    
    @model
    def create_new_itinerary(self, user_id: str, preferences: UserPreference) -> TravelItinerary:
        """Create a new travel itinerary based on user preferences"""
        pass  # Implemented by PydanticAI
    
    @model
    def update_itinerary(self, itinerary: TravelItinerary, user_input: str, preferences: UserPreference) -> TravelItinerary:
        """Update an existing travel itinerary based on user input and preferences"""
        pass  # Implemented by PydanticAI

class BookingAgent(Agent):
    """Agent responsible for managing reservations and bookings"""
    
    def __call__(self, state: GraphState) -> GraphState:
        """Make or update bookings based on the travel itinerary"""
        if not state.travel_itinerary:
            # No itinerary to book
            state.errors.append("No itinerary available for booking")
            return state
        
        # Process bookings for accommodations, transportation, and activities
        booking_results = self.process_bookings(state.travel_itinerary)
        
        # Update itinerary with booking references
        for booking in booking_results:
            self.update_itinerary_with_booking(state.travel_itinerary, booking)
        
        # Update team memory with booking confirmations
        state.team_memory.booking_history.extend(booking_results)
        
        # Create agent message
        message = AgentMessage(
            message_id=f"msg_{len(state.messages) + 1}",
            sender="booking_agent",
            recipient="user_interaction_agent",
            timestamp=datetime.now(),
            message_type="booking_update",
            content={"bookings": booking_results},
            requires_response=True
        )
        state.messages.append(message)
        
        # Set monitoring needed
        state.monitoring_needed = True
        
        return state
    
    @model
    def process_bookings(self, itinerary: TravelItinerary) -> List[BookingConfirmation]:
        """Process bookings for accommodations, transportation, and activities"""
        pass  # Implemented by PydanticAI
    
    def update_itinerary_with_booking(self, itinerary: TravelItinerary, booking: BookingConfirmation):
        """Update the itinerary with booking confirmation details"""
        # Find the corresponding item in the itinerary and update it
        for item_list in [itinerary.accommodations, itinerary.transportations, itinerary.activities]:
            for item in item_list:
                if (hasattr(item, 'booking_reference') and 
                    (item.booking_reference is None or item.booking_reference == "")):
                    # Update this item with booking details
                    item.booking_reference = booking.confirmation_code
                    item.booking_status = booking.booking_status
                    break

class MonitoringAgent(Agent):
    """Agent responsible for monitoring price changes, cancellations, and other events"""
    
    def __call__(self, state: GraphState) -> GraphState:
        """Monitor bookings and itinerary for changes, alerts, or opportunities"""
        if not state.travel_itinerary:
            # No itinerary to monitor
            return state
        
        # Monitor for price changes, cancellations, etc.
        alerts = self.monitor_bookings(
            itinerary=state.travel_itinerary,
            booking_history=state.team_memory.booking_history
        )
        
        # Add alerts to team memory
        if alerts:
            state.team_memory.monitoring_alerts.extend(alerts)
            
            # Create agent message for each alert
            for alert in alerts:
                message = AgentMessage(
                    message_id=f"msg_{len(state.messages) + 1}",
                    sender="monitoring_agent",
                    recipient="user_interaction_agent",
                    timestamp=datetime.now(),
                    message_type="alert",
                    content={"alert": alert},
                    requires_response=True
                )
                state.messages.append(message)
        
        # Task complete
        state.task_complete = True
        
        return state
    
    @model
    def monitor_bookings(self, itinerary: TravelItinerary, booking_history: List[BookingConfirmation]) -> List[MonitoringAlert]:
        """Monitor bookings for price changes, cancellations, or other issues"""
        pass  # Implemented by PydanticAI

# =============================================
# Tool Integrations
# =============================================

class AmadeusAPI(Tool):
    """Integration with Amadeus API for flight and hotel bookings"""
    
    @model
    def search_flights(self, origin: str, destination: str, date: datetime) -> List[Dict[str, Any]]:
        """Search for flights between two locations on a specific date"""
        pass  # Implemented by PydanticAI
    
    @model
    def search_hotels(self, location: str, check_in: datetime, check_out: datetime) -> List[Dict[str, Any]]:
        """Search for hotels in a location for specific dates"""
        pass  # Implemented by PydanticAI
    
    @model
    def book_flight(self, flight_id: str, passenger_details: Dict[str, Any]) -> BookingConfirmation:
        """Book a flight"""
        pass  # Implemented by PydanticAI
    
    @model
    def book_hotel(self, hotel_id: str, room_type: str, guest_details: Dict[str, Any]) -> BookingConfirmation:
        """Book a hotel"""
        pass  # Implemented by PydanticAI

class SabreAPI(Tool):
    """Integration with Sabre API for travel services"""
    
    @model
    def search_packages(self, origin: str, destination: str, date_range: Dict[str, datetime]) -> List[Dict[str, Any]]:
        """Search for travel packages"""
        pass  # Implemented by PydanticAI
    
    @model
    def book_package(self, package_id: str, traveler_details: Dict[str, Any]) -> BookingConfirmation:
        """Book a travel package"""
        pass  # Implemented by PydanticAI

class RentalcarsAPI(Tool):
    """Integration with Rentalcars API for car rentals"""
    
    @model
    def search_cars(self, location: str, pickup_date: datetime, return_date: datetime) -> List[Dict[str, Any]]:
        """Search for rental cars"""
        pass  # Implemented by PydanticAI
    
    @model
    def book_car(self, car_id: str, renter_details: Dict[str, Any]) -> BookingConfirmation:
        """Book a rental car"""
        pass  # Implemented by PydanticAI

class KaibanJS(Tool):
    """Integration with KaibanJS for specialized automation"""
    
    @model
    def optimize_itinerary(self, itinerary: TravelItinerary) -> TravelItinerary:
        """Optimize an itinerary for efficiency"""
        pass  # Implemented by PydanticAI
    
    @model
    def generate_trip_summary(self, itinerary: TravelItinerary) -> str:
        """Generate a user-friendly summary of a trip"""
        pass  # Implemented by PydanticAI

# =============================================
# LangGraph Workflow Definition
# =============================================

def create_multi_agent_graph():
    """Create the LangGraph workflow for the multi-agent system"""
    
    # Initialize agents
    user_interaction_agent = UserInteractionAgent()
    planning_agent = PlanningAgent()
    booking_agent = BookingAgent()
    monitoring_agent = MonitoringAgent()
    
    # Create state graph
    workflow = StateGraph(GraphState)
    
    # Add agents as nodes
    workflow.add_node("user_interaction", user_interaction_agent)
    workflow.add_node("planning", planning_agent)
    workflow.add_node("booking", booking_agent)
    workflow.add_node("monitoring", monitoring_agent)
    
    # Define edges (connections between agents)
    workflow.add_edge("user_interaction", "planning", condition=lambda x: x.planning_needed)
    workflow.add_edge("planning", "booking", condition=lambda x: x.booking_needed)
    workflow.add_edge("booking", "monitoring", condition=lambda x: x.monitoring_needed)
    workflow.add_edge("monitoring", "user_interaction", condition=lambda x: not x.task_complete)
    workflow.add_edge("monitoring", END, condition=lambda x: x.task_complete)
    
    # Conditional returns to user interaction
    workflow.add_edge("planning", "user_interaction", 
                      condition=lambda x: not x.booking_needed)
    workflow.add_edge("booking", "user_interaction", 
                      condition=lambda x: not x.monitoring_needed)
    
    # Define entry point
    workflow.set_entry_point("user_interaction")
    
    return workflow

# =============================================
# Data Encryption & Security
# =============================================

class DataEncryption:
    """Handles data encryption and security for sensitive information"""
    
    @staticmethod
    def encrypt_user_data(data: Any) -> str:
        """Encrypt user data before storage"""
        # Implementation would use a proper encryption library
        return f"encrypted_{str(data)}"
    
    @staticmethod
    def decrypt_user_data(encrypted_data: str) -> Any:
        """Decrypt user data for processing"""
        # Implementation would use a proper decryption method
        if encrypted_data.startswith("encrypted_"):
            return encrypted_data[10:]
        return encrypted_data
    
    @staticmethod
    def anonymize_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize personal data for compliance"""
        # Implementation would properly anonymize PII
        anonymized = data.copy()
        if "name" in anonymized:
            anonymized["name"] = "ANONYMIZED"
        if "email" in anonymized:
            anonymized["email"] = "ANONYMIZED"
        return anonymized

# =============================================
# Main System Implementation
# =============================================

class TravelPlanningSystem:
    """Main system class that integrates all components"""
    
    def __init__(self):
        """Initialize the system"""
        self.workflow = create_multi_agent_graph()
        self.team_memory = TeamMemory(user_profile=UserPreference())
        
        # Initialize tools
        self.tools = {
            "amadeus": AmadeusAPI(),
            "sabre": SabreAPI(),
            "rentalcars": RentalcarsAPI(),
            "kaiban": KaibanJS()
        }
        
        # Security
        self.security = DataEncryption()
    
    def process_user_request(self, user_id: str, user_input: str) -> Dict[str, Any]:
        """Process a user request through the multi-agent system"""
        # Initialize state
        state = GraphState(
            user_id=user_id,
            user_input=user_input,
            team_memory=self.team_memory
        )
        
        # Run the workflow
        result = self.workflow.run(state)
        
        # Update team memory with the final state
        self.team_memory = result.team_memory
        
        # Return response to user
        return {
            "itinerary": result.travel_itinerary,
            "messages": [msg for msg in result.messages if msg.recipient == "user_interaction_agent"],
            "alerts": result.team_memory.monitoring_alerts
        }

# Example usage
if __name__ == "__main__":
    system = TravelPlanningSystem()
    result = system.process_user_request(
        user_id="user123",
        user_input="I want to plan a trip to Paris for next week with my family. We're interested in cultural activities and prefer to stay in a hotel."
    )
    print(json.dumps(result, indent=2))