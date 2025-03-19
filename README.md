# TravelGraph: Multi-Agent Travel Planning System

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![LangGraph](https://img.shields.io/badge/langgraph-0.0.33%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

A production-ready AI system for autonomous travel planning, combining specialized agents, LangGraph orchestration, PydanticAI data validation, and a continuous learning reinforcement framework.

![TravelGraph System Architecture](docs/images/system_architecture.png)

## 🔑 Key Features

- **Multi-Agent Architecture**: Specialized agents collaborate to handle different aspects of travel planning
- **LLM Provider Diversity**: Support for different LLM providers (OpenAI, Anthropic, Azure) for each agent
- **Continuous Learning**: RL framework optimizes agent performance over time based on feedback
- **Structured Knowledge**: PydanticAI ensures type safety and valid data across the entire system
- **Web UI**: User-friendly interface for interacting with the agent team
- **Monitoring**: Prometheus and Grafana integration for performance monitoring
- **Dockerized**: Easy deployment with Docker Compose

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- API keys for at least one LLM provider (OpenAI, Anthropic)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/travel-agent-team.git
   cd travel-agent-team
   ```

2. Configure API keys:
   ```bash
   cp .env.example .env
   # Edit .env to add your OpenAI and/or Anthropic API keys
   ```

3. Start the services:
   ```bash
   chmod +x start.sh
   ./start.sh
   ```

4. Access the Web UI:
   - URL: http://localhost:8000
   - Default login: `admin` / `travel123`

### Security Note

For production deployment, make sure to:
- Change the default API credentials in the `.env` file
- Use secure values for all passwords
- Configure proper HTTPS for all services

## 🧠 System Components

### Multi-Agent Team

| Agent | Responsibility | Learning Focus |
|-------|---------------|---------------|
| **User Interaction Agent** | Understands user requests and manages conversations | Conversation flow optimization |
| **Planning Agent** | Creates and modifies travel itineraries | Preference matching, activity selection |
| **Booking Agent** | Manages reservations across travel services | Price optimization, booking success |
| **Monitoring Agent** | Tracks price changes and booking status | Alert prioritization, issue detection |

### LLM Provider Diversity

Each agent can use a different LLM provider:
- OpenAI (GPT-3.5/4/4o)
- Anthropic (Claude models)
- Azure OpenAI

This enables diversity of thought, optimization of costs, and resilience against provider outages.

### Services

- **Web UI**: User interface for planning trips and managing configurations
- **API Server**: Flask-based API with authentication
- **Prometheus**: Metrics collection
- **Grafana**: Dashboards and visualizations
- **Node Exporter**: Host metrics
- **cAdvisor**: Container metrics

## 💻 Development

<<<<<<< HEAD
### Reinforcement Learning Framework

![RL Architecture](https://github.com/cgordon-dev/travel-agent-team/raw/main/docs/images/rl_architecture.png)

The RL system provides continuous improvement through:

1. **Experience Collection**
   - Captures state-action-reward-next_state tuples
   - Persistent storage with efficient memory management
   - Handles both implicit & explicit feedback

2. **Agent-Specific Reward Functions**
   - User satisfaction metrics for interaction quality
   - Preference matching for itinerary planning
   - Booking efficiency & cost optimization 
   - Monitoring effectiveness for detecting issues

3. **Policy Optimization Mechanisms**
   - Parameter tuning based on performance
   - Cross-user learning for generalization
   - A/B testing of agent strategies

### Production Infrastructure

- **Configurable System**: JSON-based configuration for all parameters
- **Persistence Layer**: Stores experiences, models, and user profiles
- **Web Service Integration**: Ready for deployment with production APIs
- **Monitoring & Analytics**: Performance tracking and system insights
- **Security Features**: Data encryption and privacy compliance

## 📂 Project Structure
=======
### Project Structure
>>>>>>> ad2ad87 (added monitoring scripts and setup via prometheus and grafana)

```
travel-agent-team/
├── travel_agent_architecture.py  # Core multi-agent system
├── pydantic_models.py            # Data validation models
├── reinforcement_learning.py     # RL framework components
├── api_server.py                 # API and Web UI server
├── llm_config.py                 # LLM provider configuration
├── ui/                           # Web interface
├── docker-compose.yaml           # Docker deployment config
├── prometheus/                   # Prometheus configuration
├── grafana/                      # Grafana dashboards
└── requirements.txt              # Dependencies
```

### Environment Variables

The following environment variables can be set in the `.env` file:

```
# LLM Provider API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# API Authentication
API_USERNAME=admin
API_PASSWORD=secure_password_here

# Azure OpenAI (Optional)
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_VERSION=2023-05-15
AZURE_OPENAI_DEPLOYMENT_FAST=your-gpt35-deployment-name
AZURE_OPENAI_DEPLOYMENT_BALANCED=your-gpt4-deployment-name
AZURE_OPENAI_DEPLOYMENT_POWERFUL=your-gpt4o-deployment-name
```

### Running Tests

```bash
<<<<<<< HEAD
# Clone the repository
git clone https://github.com/cgordon-dev/travel-agent-team.git
cd travel-agent-team
=======
# Run unit tests
pytest tests/
>>>>>>> ad2ad87 (added monitoring scripts and setup via prometheus and grafana)

# Run API tests
pytest tests/test_api.py
```

## 📊 Monitoring

### Metrics

The system exposes Prometheus metrics at http://localhost:8001, including:

- Request counts and latencies
- Error rates
- LLM token usage by provider
- Agent activity metrics
- User statistics

### Dashboards

Grafana dashboards are available at http://localhost:3000 (login: admin / travel_agent_admin):

- System Overview
- LLM Provider Statistics
- Reinforcement Learning Metrics
- User Activity

## 📚 Documentation

- [Docker Setup](DOCKER_README.md)
- [LLM Configuration](LLM_CONFIG_README.md)
- [API Documentation](docs/API.md)
- [Reinforcement Learning Framework](docs/RL_FRAMEWORK.md)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<<<<<<< HEAD
## 📚 Citation

If you use this code in your research, please cite:

```
@software{travelgraph2023,
  author = {Carl Gordon},
  title = {TravelGraph: Multi-Agent Travel Planning System with Reinforcement Learning},
  year = {2023},
  url = {https://github.com/cgordon-dev/travel-agent-team}
}
```

=======
>>>>>>> ad2ad87 (added monitoring scripts and setup via prometheus and grafana)
## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for LangGraph
- [Pydantic](https://github.com/pydantic/pydantic) for data validation
<<<<<<< HEAD
- The reinforcement learning community for algorithms and best practices
=======
- [Flask](https://flask.palletsprojects.com/) for the web server
- [Prometheus](https://prometheus.io/) and [Grafana](https://grafana.com/) for monitoring
>>>>>>> ad2ad87 (added monitoring scripts and setup via prometheus and grafana)
