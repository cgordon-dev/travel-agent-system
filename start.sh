#!/bin/bash

# Start the Travel Agent system with Docker Compose

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file not found!"
    echo "Please create a .env file with your API keys first."
    echo "You can copy .env.example and fill in your API keys."
    exit 1
fi

# Create necessary directories
mkdir -p data rl_data rl_models

# Start services
docker-compose up -d

echo "\nTravel Agent services started successfully!\n"
echo "API Server:     http://localhost:8000/api/v1/travel/request"
echo "Metrics:       http://localhost:8001"
echo "Prometheus:    http://localhost:9090"
echo "Grafana:       http://localhost:3000"
echo "\nGrafana login: admin / travel_agent_admin"
