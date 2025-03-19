# TravelGraph API Documentation

The TravelGraph system exposes several REST API endpoints that can be used to interact with the multi-agent travel planning system.

## Authentication

All API endpoints are secured with HTTP Basic Authentication.

**Default credentials**:
- Username: `admin`
- Password: `travel123` (should be changed in production)

You can set custom credentials in your `.env` file:
```
API_USERNAME=your_custom_username
API_PASSWORD=your_secure_password
```

## Endpoints

### Health Check

```
GET /health
```

Check if the API server is running.

**Response**:
```json
{
  "status": "ok",
  "timestamp": "2023-07-21T14:25:10.123456"
}
```

### Travel Request

```
POST /api/v1/travel/request
```

Submit a travel planning request to the multi-agent system.

**Request Body**:
```json
{
  "user_id": "user123",
  "user_input": "I want to plan a weekend trip to San Francisco next month with my family. We're interested in cultural activities and seafood restaurants."
}
```

**Response**:
```json
{
  "itinerary": {
    "itinerary_id": "trip_12345",
    "title": "Weekend Family Trip to San Francisco",
    "start_date": "2023-08-15T00:00:00",
    "end_date": "2023-08-17T00:00:00",
    "accommodations": [...],
    "transportations": [...],
    "activities": [...],
    "total_cost": 1500,
    "currency": "USD",
    "status": "draft"
  },
  "messages": [...],
  "alerts": [...]
}
```

### Feedback Submission

```
POST /api/v1/travel/feedback
```

Submit feedback for a travel planning request, which is used by the reinforcement learning system to improve agent performance.

**Request Body**:
```json
{
  "user_id": "user123",
  "feedback": {
    "satisfaction_score": 4.5,
    "comments": "Great suggestions, but I'd like more family-friendly activities.",
    "itinerary_rating": 4,
    "trip_id": "trip_12345"
  }
}
```

**Response**:
```json
{
  "status": "Feedback recorded"
}
```

### LLM Configuration (GET)

```
GET /api/v1/system/llm-config
```

Get the current LLM configuration for all agents.

**Response**:
```json
{
  "available_providers": ["openai", "anthropic"],
  "agent_providers": {
    "user_interaction_agent": "anthropic",
    "planning_agent": "openai",
    "booking_agent": "openai",
    "monitoring_agent": "anthropic"
  },
  "agent_model_types": {
    "user_interaction_agent": "powerful",
    "planning_agent": "powerful",
    "booking_agent": "balanced",
    "monitoring_agent": "balanced"
  }
}
```

### LLM Configuration (PUT)

```
PUT /api/v1/system/llm-config
```

Update the LLM provider for a specific agent.

**Request Body**:
```json
{
  "agent_name": "planning_agent",
  "provider": "anthropic"
}
```

**Response**:
```json
{
  "status": "Configuration updated",
  "config": {
    "available_providers": ["openai", "anthropic"],
    "agent_providers": {
      "user_interaction_agent": "anthropic",
      "planning_agent": "anthropic",
      "booking_agent": "openai",
      "monitoring_agent": "anthropic"
    },
    "agent_model_types": {...}
  }
}
```

### System Stats

```
GET /api/v1/system/stats
```

Get system statistics including user activity, request counts, and RL metrics.

**Response**:
```json
{
  "active_users": 5,
  "total_requests": 42,
  "error_rate": 0.02,
  "llm_tokens": {
    "openai": 12500,
    "anthropic": 15800
  },
  "rl_metrics": {
    "training_metrics": {
      "training_counts": {
        "user_interaction_agent": 15,
        "planning_agent": 12,
        "booking_agent": 10,
        "monitoring_agent": 8
      },
      "avg_rewards": {
        "user_interaction_agent": 0.82,
        "planning_agent": 0.75,
        "booking_agent": 0.68,
        "monitoring_agent": 0.71
      }
    }
  }
}
```

## Client Usage Examples

### Python

```python
import requests
from requests.auth import HTTPBasicAuth

# API configuration
BASE_URL = "http://localhost:8000"
AUTH = HTTPBasicAuth("admin", "travel123")

# Submit a travel request
response = requests.post(
    f"{BASE_URL}/api/v1/travel/request",
    json={
        "user_id": "user123",
        "user_input": "I want to plan a weekend trip to San Francisco"
    },
    auth=AUTH
)

if response.status_code == 200:
    trip_data = response.json()
    print(f"Trip created: {trip_data['itinerary']['title']}")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

### JavaScript/Fetch

```javascript
// API configuration
const BASE_URL = 'http://localhost:8000';
const API_AUTH = {
    username: 'admin',
    password: 'travel123'
};

// Helper function for authentication
function createFetchOptions(method = 'GET', body = null) {
    const options = {
        method,
        headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Basic ' + btoa(`${API_AUTH.username}:${API_AUTH.password}`)
        }
    };
    
    if (body) {
        options.body = JSON.stringify(body);
    }
    
    return options;
}

// Submit a travel request
async function submitTravelRequest() {
    try {
        const response = await fetch(`${BASE_URL}/api/v1/travel/request`, 
            createFetchOptions('POST', {
                user_id: 'user123',
                user_input: 'I want to plan a weekend trip to San Francisco'
            })
        );
        
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }
        
        const data = await response.json();
        console.log(`Trip created: ${data.itinerary.title}`);
    } catch (error) {
        console.error('Error:', error);
    }
}
```

### cURL

```bash
# Submit a travel request
curl -X POST http://localhost:8000/api/v1/travel/request \
  -H "Content-Type: application/json" \
  -u admin:travel123 \
  -d '{
    "user_id": "user123",
    "user_input": "I want to plan a weekend trip to San Francisco"
  }'

# Get LLM configuration
curl -X GET http://localhost:8000/api/v1/system/llm-config \
  -u admin:travel123

# Update LLM configuration
curl -X PUT http://localhost:8000/api/v1/system/llm-config \
  -H "Content-Type: application/json" \
  -u admin:travel123 \
  -d '{
    "agent_name": "planning_agent",
    "provider": "anthropic"
  }'
```