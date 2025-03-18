"""
Core Pydantic Models for Travel & Event Planning System
These models demonstrate the structured data approach with PydanticAI
"""

from typing import List, Dict, Any, Optional, Union, Literal
from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel, Field, validator, ConfigDict, model_validator

# =============================================
# Basic Data Models
# =============================================

class Location(BaseModel):
    """Geographical location model with optional coordinates"""
    city: str
    country: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    
    @model_validator(mode="after")
    def validate_coordinates(self):
        """Ensure coordinates are valid if provided"""
        if self.latitude is not None and (self.latitude < -90 or self.latitude > 90):
            raise ValueError("Latitude must be between -90 and 90")
        if self.longitude is not None and (self.longitude < -180 or self.longitude > 180):
            raise ValueError("Longitude must be between -180 and 180")
        return self

class PriceRange(BaseModel):
    """Model for price ranges with currency"""
    min_price: float
    max_price: float
    currency: str = "USD"
    
    @model_validator(mode="after")
    def validate_price_range(self):
        """Ensure min_price is less than max_price"""
        if self.min_price > self.max_price:
            raise ValueError("min_price must be less than or equal to max_price")
        if self.min_price < 0:
            raise ValueError("min_price cannot be negative")
        return self

# =============================================
# Enum Definitions for Structured Categories
# =============================================

class AccommodationType(str, Enum):
    """Types of accommodations for structured selection"""
    HOTEL = "hotel"
    APARTMENT = "apartment"
    HOSTEL = "hostel"
    RESORT = "resort"
    HOUSE = "house"

class TransportationType(str, Enum):
    """Types of transportation for structured selection"""
    FLIGHT = "flight"
    TRAIN = "train"
    BUS = "bus"
    CAR_RENTAL = "car_rental"
    FERRY = "ferry"

class ActivityType(str, Enum):
    """Types of activities for structured selection"""
    SIGHTSEEING = "sightseeing"
    CULTURAL = "cultural"
    ADVENTURE = "adventure"
    CULINARY = "culinary"
    RELAXATION = "relaxation"
    ENTERTAINMENT = "entertainment"

class AlertSeverity(str, Enum):
    """Severity levels for monitoring alerts"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# =============================================
# User Preference Model with Schema Evolution
# =============================================

class UserPreference(BaseModel):
    """
    Comprehensive user preference model with dynamic schema evolution
    This model can be extended at runtime with new preference types
    """
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
    climate_preferences: Optional[List[str]] = None
    shopping_interests: Optional[List[str]] = None
    
    # Dynamic preference learning
    learned_preferences: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dynamically updated preferences based on user feedback and interactions"
    )
    preference_confidence_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Confidence scores for learned preferences (0.0-1.0)"
    )
    
    # Schema evolution helper method
    @classmethod
    def update_schema(cls, new_fields: Dict[str, Dict[str, Any]]):
        """
        Dynamically extend the schema with new fields as agents learn new preference types
        
        Args:
            new_fields: Dictionary mapping field names to their metadata
                Expected format:
                {
                    "field_name": {
                        "type": type,
                        "default": default_value,
                        "description": "Field description"
                    }
                }
        """
        for field_name, field_info in new_fields.items():
            if field_name not in cls.__annotations__:
                # Add new field to the model
                cls.__annotations__[field_name] = field_info["type"]
                # Add default value
                default_value = field_info.get("default", None)
                setattr(cls, field_name, default_value)
    
    def update_preference(self, category: str, value: Any, confidence: float = 0.5):
        """
        Update a learned preference with a confidence score
        
        Args:
            category: The preference category
            value: The preference value
            confidence: Confidence score between 0.0 and 1.0
        """
        self.learned_preferences[category] = value
        self.preference_confidence_scores[category] = max(0.0, min(1.0, confidence))
    
    def merge_preferences(self, new_preferences: Dict[str, Any], confidence_threshold: float = 0.7):
        """
        Merge new learned preferences if they meet the confidence threshold
        
        Args:
            new_preferences: Dictionary of new preferences
            confidence_threshold: Minimum confidence score to accept new preferences
        """
        for category, value in new_preferences.items():
            confidence = new_preferences.get(f"{category}_confidence", 0.5)
            if confidence >= confidence_threshold:
                self.update_preference(category, value, confidence)

# =============================================
# Travel Itinerary Models with Inheritance
# =============================================

class TravelItineraryItem(BaseModel):
    """Base class for all itinerary items with common fields"""
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
    cancellation_deadline: Optional[datetime] = None
    
    @model_validator(mode="after")
    def validate_times(self):
        """Ensure start_time is before end_time if both are provided"""
        if self.start_time and self.end_time and self.start_time > self.end_time:
            raise ValueError("start_time must be before end_time")
        return self

class Accommodation(TravelItineraryItem):
    """Accommodation details with specific fields"""
    accommodation_type: AccommodationType
    check_in_time: datetime
    check_out_time: datetime
    amenities: List[str] = Field(default_factory=list)
    room_type: Optional[str] = None
    number_of_guests: int = 1
    
    @model_validator(mode="after")
    def validate_accommodation_times(self):
        """Ensure check_in_time is before check_out_time"""
        if self.check_in_time > self.check_out_time:
            raise ValueError("check_in_time must be before check_out_time")
        return self
    
class Transportation(TravelItineraryItem):
    """Transportation details with specific fields"""
    transportation_type: TransportationType
    departure_location: Location
    arrival_location: Location
    operator: Optional[str] = None
    seat_details: Optional[str] = None
    confirmation_number: Optional[str] = None
    
    @model_validator(mode="after")
    def validate_locations(self):
        """Ensure departure and arrival locations are different"""
        if (self.departure_location.city == self.arrival_location.city and 
            self.departure_location.country == self.arrival_location.country):
            raise ValueError("departure_location and arrival_location must be different")
        return self
    
class Activity(TravelItineraryItem):
    """Activity details with specific fields"""
    activity_type: ActivityType
    duration: timedelta
    included_items: List[str] = Field(default_factory=list)
    meeting_point: Optional[str] = None
    group_size: Optional[int] = None
    physical_intensity: Optional[str] = None
    
    @model_validator(mode="after")
    def validate_duration(self):
        """Ensure duration is positive"""
        if self.duration.total_seconds() <= 0:
            raise ValueError("duration must be positive")
        return self

class TravelItinerary(BaseModel):
    """
    Complete travel itinerary model that organizes accommodations, 
    transportation, and activities
    """
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
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    
    @model_validator(mode="after")
    def validate_dates(self):
        """Ensure start_date is before end_date"""
        if self.start_date > self.end_date:
            raise ValueError("start_date must be before end_date")
        return self
    
    def add_item(self, item: Union[Accommodation, Transportation, Activity]):
        """
        Add an item to the appropriate list based on its type and update total cost
        
        Args:
            item: The itinerary item to add
        """
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
        
        # Update last_updated timestamp
        self.last_updated = datetime.now()
    
    def get_daily_schedule(self, day: int) -> List[TravelItineraryItem]:
        """
        Get all items scheduled for a specific day
        
        Args:
            day: The day number to retrieve items for
            
        Returns:
            List of itinerary items for the day, sorted by start time
        """
        daily_items = []
        
        for item_list in [self.accommodations, self.transportations, self.activities]:
            for item in item_list:
                if item.day == day:
                    daily_items.append(item)
        
        # Sort by start time
        return sorted(daily_items, key=lambda x: x.start_time or datetime.min)
    
    def get_price_breakdown(self) -> Dict[str, float]:
        """
        Get a breakdown of prices by category
        
        Returns:
            Dictionary with category totals
        """
        return {
            "accommodations": sum(a.price or 0 for a in self.accommodations),
            "transportation": sum(t.price or 0 for t in self.transportations),
            "activities": sum(a.price or 0 for a in self.activities)
        }

# =============================================
# Booking and Monitoring Models
# =============================================

class BookingConfirmation(BaseModel):
    """Model for booking confirmations with provider details"""
    booking_id: str
    service_type: str
    provider: str
    booking_status: str
    confirmation_code: str
    booking_date: datetime
    price: float
    currency: str = "USD"
    cancellation_policy: Optional[str] = None
    refundable: bool = True
    cancellation_deadline: Optional[datetime] = None
    details: Dict[str, Any] = Field(default_factory=dict)

class MonitoringAlert(BaseModel):
    """Model for monitoring alerts with recommended actions"""
    alert_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    alert_type: str
    severity: AlertSeverity
    affected_booking_id: Optional[str] = None
    description: str
    recommended_action: Optional[str] = None
    price_change: Optional[float] = None
    deadline_timestamp: Optional[datetime] = None
    resolved: bool = False
    
    def resolve(self, resolution_notes: Optional[str] = None):
        """
        Mark the alert as resolved
        
        Args:
            resolution_notes: Optional notes about how the alert was resolved
        """
        self.resolved = True
        if resolution_notes:
            self.details = self.details or {}
            self.details["resolution_notes"] = resolution_notes
            self.details["resolved_at"] = datetime.now()

# =============================================
# Agent Communication Models
# =============================================

class AgentMessageType(str, Enum):
    """Types of messages that can be exchanged between agents"""
    INTENT_ANALYSIS = "intent_analysis"
    ITINERARY_UPDATE = "itinerary_update"
    BOOKING_REQUEST = "booking_request"
    BOOKING_CONFIRMATION = "booking_confirmation"
    BOOKING_ERROR = "booking_error"
    PRICE_ALERT = "price_alert"
    CANCELLATION_ALERT = "cancellation_alert"
    USER_RECOMMENDATION = "user_recommendation"
    ERROR = "error"
    STATUS_UPDATE = "status_update"

class AgentMessage(BaseModel):
    """Model for structured communication between agents"""
    message_id: str
    sender: str
    recipient: str
    timestamp: datetime = Field(default_factory=datetime.now)
    message_type: AgentMessageType
    content: Dict[str, Any]
    requires_response: bool = False
    
    def create_response(self, content: Dict[str, Any], message_type: AgentMessageType) -> "AgentMessage":
        """
        Create a response message to this message
        
        Args:
            content: The content for the response
            message_type: The type of the response message
            
        Returns:
            A new AgentMessage instance configured as a response
        """
        return AgentMessage(
            message_id=f"response-to-{self.message_id}",
            sender=self.recipient,
            recipient=self.sender,
            message_type=message_type,
            content=content,
            requires_response=False
        )

# =============================================
# Team Memory Model
# =============================================

class TeamMemory(BaseModel):
    """
    Shared memory model that all agents can access for common knowledge and context
    Serves as the collective intelligence and state repository for the agent team
    """
    user_profile: UserPreference
    current_itinerary: Optional[TravelItinerary] = None
    booking_history: List[BookingConfirmation] = Field(default_factory=list)
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    monitoring_alerts: List[MonitoringAlert] = Field(default_factory=list)
    shared_context: Dict[str, Any] = Field(default_factory=dict)
    
    def update_memory(self, key: str, value: Any):
        """
        Update a specific section of the team memory
        
        Args:
            key: The memory section to update
            value: The new value for the section
        """
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self.shared_context[key] = value
    
    def add_conversation_entry(self, role: str, content: str):
        """
        Add an entry to the conversation history
        
        Args:
            role: The role of the speaker (user, system, agent name)
            content: The content of the message
        """
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now()
        })
    
    def add_alert(self, alert: MonitoringAlert):
        """
        Add a monitoring alert and sort by severity
        
        Args:
            alert: The alert to add
        """
        self.monitoring_alerts.append(alert)
        # Sort alerts: unresolved first, then by severity, then by timestamp
        self.monitoring_alerts.sort(
            key=lambda a: (
                a.resolved, 
                -["low", "medium", "high", "critical"].index(a.severity), 
                a.timestamp
            )
        )
    
    def get_recent_conversations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most recent conversation entries
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of recent conversation entries
        """
        return self.conversation_history[-limit:]
    
    def get_active_alerts(self) -> List[MonitoringAlert]:
        """
        Get all unresolved alerts
        
        Returns:
            List of unresolved alerts
        """
        return [alert for alert in self.monitoring_alerts if not alert.resolved]