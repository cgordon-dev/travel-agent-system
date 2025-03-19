# LLM Configuration for Travel Agent System

This document explains how to configure the Travel Agent System to use different LLM providers for different agents, enabling diversity of thought in the multi-agent system.

## Supported LLM Providers

The system supports the following LLM providers:

- **OpenAI** - GPT-3.5, GPT-4, GPT-4o models
- **Anthropic** - Claude models (Haiku, Sonnet, Opus)
- **Azure OpenAI** - Azure-hosted OpenAI models

## Configuration

### API Keys

To use the LLM providers, you need to set up API keys in your `.env` file:

1. Copy the `.env.example` file to a new file named `.env`
2. Add your API keys to the `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

You need to provide at least one of these API keys for the system to function. If both are provided, you can freely switch between providers.

### Azure OpenAI Configuration

If you want to use Azure OpenAI, you need to provide additional configuration:

```
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_VERSION=2023-05-15
AZURE_OPENAI_DEPLOYMENT_FAST=your-gpt35-deployment-name
AZURE_OPENAI_DEPLOYMENT_BALANCED=your-gpt4-deployment-name
AZURE_OPENAI_DEPLOYMENT_POWERFUL=your-gpt4o-deployment-name
```

## Default Configuration

By default, the system uses the following configuration:

| Agent | Provider | Model Type |
|-------|----------|------------|
| user_interaction_agent | Anthropic | Powerful (Claude-3-Opus) |
| planning_agent | OpenAI | Powerful (GPT-4o) |
| booking_agent | OpenAI | Balanced (GPT-4o-mini) |
| monitoring_agent | Anthropic | Balanced (Claude-3-Haiku) |

This configuration provides a good balance of capabilities and diversity of thought across the agents.

## Customizing LLM Providers

You can customize which LLM provider each agent uses in your code:

```python
from travel_agent_architecture import TravelPlanningSystem
from llm_config import LLMProvider, ModelType

# Create a configuration that uses different providers
config_override = {
    "agent_providers": {
        "user_interaction_agent": LLMProvider.ANTHROPIC,
        "planning_agent": LLMProvider.OPENAI,
        "booking_agent": LLMProvider.OPENAI,
        "monitoring_agent": LLMProvider.ANTHROPIC
    },
    "agent_model_types": {
        "user_interaction_agent": ModelType.POWERFUL,
        "planning_agent": ModelType.POWERFUL,
        "booking_agent": ModelType.BALANCED,
        "monitoring_agent": ModelType.BALANCED
    }
}

# Initialize the system with our configuration
system = TravelPlanningSystem(llm_config_override=config_override)
```

## Dynamically Switching Providers

You can also switch providers dynamically during runtime:

```python
# Switch the planning agent to use Anthropic
system.configure_agent_provider("planning_agent", LLMProvider.ANTHROPIC)

# Get the current configuration
current_config = system.get_current_llm_configuration()
print(current_config)
```

## Example Usage

See `llm_example.py` for a complete example of:

1. Using a single provider for all agents
2. Using different providers for different agents
3. Dynamically switching providers during runtime

## Benefits of Provider Diversity

Using different LLM providers for different agents has several benefits:

1. **Diversity of thought** - Different models approach problems in different ways
2. **Specialized capabilities** - Some models may be better at certain tasks
3. **Cost optimization** - Use more expensive models only for tasks that benefit from them
4. **Redundancy** - If one provider has an outage, some functionality remains available

## Troubleshooting

If you encounter issues with the LLM configuration:

1. Ensure your API keys are correct and have sufficient quota
2. Check that you have installed the required dependencies: `pip install -r requirements.txt`
3. Verify that your `.env` file is in the correct location (root of the project)
4. If using Azure OpenAI, ensure your deployment names match the model types