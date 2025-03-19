# Dockerized Travel Agent System

This document explains how to run the Travel Agent system using Docker and monitor it with Prometheus and Grafana.

## Prerequisites

- Docker and Docker Compose installed
- API keys for at least one LLM provider (OpenAI, Anthropic)

## Getting Started

### 1. Configure API Keys

Create a `.env` file with your API keys:

```bash
# Copy the example file
cp .env.example .env

# Edit the file to add your API keys
nano .env  # or use any text editor
```

Add your OpenAI and/or Anthropic API keys to the `.env` file.

### 2. Start the Services

Use the provided script to start all services:

```bash
# Make the script executable
chmod +x start.sh

# Start the services
./start.sh
```

This will start the following services:

- **Travel Agent Web UI**: Accessible at http://localhost:8000
  - Login with default credentials: username: `admin` / password: `travel123`
- **Travel Agent API**: Accessible at http://localhost:8000/api/v1
  - Secured with basic auth (username: `admin` / password: `travel123`)
- **Prometheus**: Accessible at http://localhost:9090
- **Grafana**: Accessible at http://localhost:3000 (login: admin / travel_agent_admin)
- **Node Exporter**: For host metrics
- **cAdvisor**: For container metrics

### 3. Stop the Services

To stop all services:

```bash
# Make the script executable
chmod +x stop.sh

# Stop the services
./stop.sh
```

## Using the API

### Make a Travel Request

```bash
curl -X POST http://localhost:8000/api/v1/travel/request \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "user_input": "I want to plan a weekend trip to San Francisco next month."
  }'
```

### Submit Feedback

```bash
curl -X POST http://localhost:8000/api/v1/travel/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "feedback": {
      "satisfaction_score": 4.5,
      "itinerary_rating": 4,
      "booking_experience": 5,
      "comments": "Great suggestions!"
    }
  }'
```

### Get System Stats

```bash
curl http://localhost:8000/api/v1/system/stats
```

### Get/Update LLM Configuration

Get current configuration:
```bash
curl http://localhost:8000/api/v1/system/llm-config
```

Update configuration:
```bash
curl -X PUT http://localhost:8000/api/v1/system/llm-config \
  -H "Content-Type: application/json" \
  -d '{
    "agent_name": "planning_agent",
    "provider": "anthropic"
  }'
```

## Monitoring

### Grafana Dashboards

Access Grafana at http://localhost:3000 and log in with:
- Username: `admin`
- Password: `travel_agent_admin`

A pre-configured dashboard is available: "Travel Agent Dashboard"

### Prometheus Metrics

Access Prometheus at http://localhost:9090

Key metrics to monitor:

- `travel_agent_requests_total`: Total API requests
- `travel_agent_requests_errors_total`: Failed API requests
- `travel_agent_request_duration_seconds`: Request latency
- `travel_agent_agent_usage_total`: Agent usage counts
- `travel_agent_llm_tokens_total`: Token usage by provider
- `travel_agent_active_users`: Active user count

### Alerts

Pre-configured alerts:

- **TravelAgentHighErrorRate**: Triggered when error rate > 10% for 5 minutes
- **TravelAgentHighLatency**: Triggered when average latency > 2s for 5 minutes
- **TravelAgentDown**: Triggered when the service is down for 1 minute

## Customization

### Adding Custom Dashboards

Add JSON dashboard definitions to the `grafana/dashboards` directory before starting the services.

### Modifying Prometheus Configuration

Edit `prometheus/prometheus.yml` to change Prometheus settings or add additional scrape targets.

### Modifying Alert Rules

Edit `prometheus/alert_rules.yml` to add or modify alert rules.

## Troubleshooting

### Checking Logs

To view logs for a specific service:

```bash
docker-compose logs travel-agent
```

To follow logs in real-time:

```bash
docker-compose logs -f travel-agent
```

### Restarting Services

To restart a specific service:

```bash
docker-compose restart travel-agent
```

### Common Issues

1. **API Keys**: If you see authentication errors, check your API keys in the `.env` file.
2. **Permissions**: Ensure the data directories have proper permissions.
3. **Port Conflicts**: If ports are already in use, modify the port mappings in `docker-compose.yaml`.
