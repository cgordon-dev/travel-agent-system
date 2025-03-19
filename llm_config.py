"""
LLM Configuration for Travel Agent System
Provides flexible LLM provider selection for PydanticAI
"""

import os
from typing import Dict, Any, Optional, List
from enum import Enum
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    
class ModelType(str, Enum):
    """Common model types across providers"""
    FAST = "fast"       # Fast, economical models
    BALANCED = "balanced"  # Balance of cost and capability
    POWERFUL = "powerful"  # Most capable models

# Default provider mapping for each agent
DEFAULT_AGENT_PROVIDERS = {
    "user_interaction_agent": LLMProvider.ANTHROPIC,
    "planning_agent": LLMProvider.OPENAI,
    "booking_agent": LLMProvider.OPENAI,
    "monitoring_agent": LLMProvider.ANTHROPIC,
}

# Default model mapping
DEFAULT_MODEL_MAPPING = {
    LLMProvider.OPENAI: {
        ModelType.FAST: "gpt-3.5-turbo",
        ModelType.BALANCED: "gpt-4o-mini",
        ModelType.POWERFUL: "gpt-4o",
    },
    LLMProvider.ANTHROPIC: {
        ModelType.FAST: "claude-instant-1.2",
        ModelType.BALANCED: "claude-3-haiku",
        ModelType.POWERFUL: "claude-3-opus",
    },
    LLMProvider.AZURE_OPENAI: {
        ModelType.FAST: "gpt-35-turbo",
        ModelType.BALANCED: "gpt-4-turbo", 
        ModelType.POWERFUL: "gpt-4o",
    }
}

# Default model type for each agent
DEFAULT_AGENT_MODEL_TYPES = {
    "user_interaction_agent": ModelType.POWERFUL,  # Understanding user intent requires high capability
    "planning_agent": ModelType.POWERFUL,         # Complex planning tasks
    "booking_agent": ModelType.BALANCED,          # Balance cost and capability
    "monitoring_agent": ModelType.BALANCED,       # Monitoring alerts
}

class LLMConfig:
    """LLM configuration manager for the travel agent system"""
    
    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        """Initialize the LLM configuration
        
        Args:
            config_override: Optional override for default configuration
        """
        # Load API keys from environment
        self.api_keys = {
            LLMProvider.OPENAI: os.getenv("OPENAI_API_KEY"),
            LLMProvider.ANTHROPIC: os.getenv("ANTHROPIC_API_KEY"),
            LLMProvider.AZURE_OPENAI: os.getenv("AZURE_OPENAI_API_KEY"),
        }
        
        # Check that required keys are available
        self._validate_api_keys()
        
        # Load Azure specific configs if using Azure
        if self.api_keys[LLMProvider.AZURE_OPENAI]:
            self.azure_config = {
                "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
                "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
                "deployment_name": {
                    ModelType.FAST: os.getenv("AZURE_OPENAI_DEPLOYMENT_FAST"),
                    ModelType.BALANCED: os.getenv("AZURE_OPENAI_DEPLOYMENT_BALANCED"),
                    ModelType.POWERFUL: os.getenv("AZURE_OPENAI_DEPLOYMENT_POWERFUL"),
                }
            }
        
        # Initialize with defaults
        self.agent_providers = DEFAULT_AGENT_PROVIDERS.copy()
        self.model_mapping = DEFAULT_MODEL_MAPPING.copy()
        self.agent_model_types = DEFAULT_AGENT_MODEL_TYPES.copy()
        
        # Apply any overrides
        if config_override:
            self._apply_config_override(config_override)
    
    def _validate_api_keys(self) -> None:
        """Validate that required API keys are available"""
        available_providers = []
        missing_providers = []
        
        for provider, key in self.api_keys.items():
            if key:
                available_providers.append(provider)
            else:
                missing_providers.append(provider)
        
        if not available_providers:
            raise ValueError(
                "No API keys found. Please set at least one of OPENAI_API_KEY, "
                "ANTHROPIC_API_KEY, or AZURE_OPENAI_API_KEY in your .env file."
            )
            
        # If any providers are missing keys, check if they're being used by default
        for agent, provider in DEFAULT_AGENT_PROVIDERS.items():
            if provider in missing_providers:
                # Reassign to an available provider
                self.agent_providers[agent] = available_providers[0]
    
    def _apply_config_override(self, config: Dict[str, Any]) -> None:
        """Apply configuration overrides
        
        Args:
            config: Configuration override dictionary
        """
        # Override agent providers
        if "agent_providers" in config:
            for agent, provider in config["agent_providers"].items():
                if provider not in self.api_keys or not self.api_keys[provider]:
                    raise ValueError(f"API key for {provider} is not available")
                self.agent_providers[agent] = provider
        
        # Override model mapping
        if "model_mapping" in config:
            for provider, models in config["model_mapping"].items():
                if provider in self.model_mapping:
                    self.model_mapping[provider].update(models)
        
        # Override agent model types
        if "agent_model_types" in config:
            self.agent_model_types.update(config["agent_model_types"])
    
    def get_agent_llm_config(self, agent_name: str) -> Dict[str, Any]:
        """Get the LLM configuration for a specific agent
        
        Args:
            agent_name: The name of the agent
            
        Returns:
            Dictionary containing LLM configuration for PydanticAI
        """
        provider = self.agent_providers.get(agent_name, LLMProvider.OPENAI)
        model_type = self.agent_model_types.get(agent_name, ModelType.BALANCED)
        model = self.model_mapping.get(provider, {}).get(model_type)
        
        if not model:
            raise ValueError(f"No model found for {provider} with type {model_type}")
        
        # Base configuration
        config = {
            "provider": provider,
            "model": model,
            "api_key": self.api_keys[provider],
        }
        
        # Add provider-specific configuration
        if provider == LLMProvider.AZURE_OPENAI:
            config.update({
                "api_version": self.azure_config["api_version"],
                "azure_endpoint": self.azure_config["azure_endpoint"],
                "deployment_name": self.azure_config["deployment_name"][model_type]
            })
        
        return config
    
    def get_pydanticai_config(self) -> Dict[str, Any]:
        """Get the full PydanticAI configuration
        
        Returns:
            Dictionary containing PydanticAI configuration
        """
        config = {
            "default_provider": LLMProvider.OPENAI,
            "default_model": self.model_mapping[LLMProvider.OPENAI][ModelType.BALANCED],
            "api_keys": {k: v for k, v in self.api_keys.items() if v},
            "agent_configs": {}
        }
        
        # Add configuration for each agent
        for agent in self.agent_providers.keys():
            config["agent_configs"][agent] = self.get_agent_llm_config(agent)
        
        return config
    
    def get_available_providers(self) -> List[LLMProvider]:
        """Get the list of available providers with valid API keys
        
        Returns:
            List of available providers
        """
        return [provider for provider, key in self.api_keys.items() if key]
    
    def switch_agent_provider(self, agent_name: str, provider: LLMProvider) -> None:
        """Switch the LLM provider for a specific agent
        
        Args:
            agent_name: The name of the agent
            provider: The provider to use
        """
        if provider not in self.api_keys or not self.api_keys[provider]:
            raise ValueError(f"API key for {provider} is not available")
        
        self.agent_providers[agent_name] = provider

# Global config instance
config = None

def initialize_llm_config(config_override: Optional[Dict[str, Any]] = None) -> LLMConfig:
    """Initialize the global LLM configuration
    
    Args:
        config_override: Optional override for default configuration
        
    Returns:
        The initialized LLM configuration
    """
    global config
    config = LLMConfig(config_override)
    return config

def get_llm_config() -> LLMConfig:
    """Get the global LLM configuration
    
    Returns:
        The global LLM configuration
    """
    global config
    if config is None:
        config = initialize_llm_config()
    return config