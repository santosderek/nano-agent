"""
Test multi-provider support for nano agent.

This test verifies that the provider configuration works correctly
for OpenAI, Anthropic, Ollama, and Azure providers.
"""

import pytest
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from nano_agent.modules.provider_config import ProviderConfig
from nano_agent.modules.constants import AVAILABLE_MODELS, PROVIDER_REQUIREMENTS
from agents import Agent, ModelSettings, OpenAIChatCompletionsModel


class TestProviderConfig:
    """Test provider configuration functionality."""
    
    def test_create_agent_openai(self):
        """Test creating an OpenAI agent."""
        with patch('nano_agent.modules.provider_config.Agent') as MockAgent:
            mock_agent = Mock()
            MockAgent.return_value = mock_agent
            
            agent = ProviderConfig.create_agent(
                name="TestAgent",
                instructions="Test instructions",
                tools=[],
                model="gpt-5-mini",
                provider="openai",
                model_settings=None
            )
            
            MockAgent.assert_called_once_with(
                name="TestAgent",
                instructions="Test instructions",
                tools=[],
                model="gpt-5-mini",
                model_settings=None
            )
            assert agent == mock_agent
    
    def test_create_agent_anthropic(self):
        """Test creating an Anthropic agent via OpenAI SDK."""
        with patch('nano_agent.modules.provider_config.Agent') as MockAgent, \
             patch('nano_agent.modules.provider_config.AsyncOpenAI') as MockAsyncOpenAI, \
             patch('nano_agent.modules.provider_config.OpenAIChatCompletionsModel') as MockModel:
            
            mock_agent = Mock()
            MockAgent.return_value = mock_agent
            mock_client = Mock()
            MockAsyncOpenAI.return_value = mock_client
            mock_model = Mock()
            MockModel.return_value = mock_model
            
            agent = ProviderConfig.create_agent(
                name="TestAgent",
                instructions="Test instructions",
                tools=[],
                model="claude-opus-4-1-20250805",
                provider="anthropic",
                model_settings=None
            )
            
            # Check that AsyncOpenAI was created with Anthropic endpoint
            MockAsyncOpenAI.assert_called_once_with(
                base_url="https://api.anthropic.com/v1/",
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
            
            # Check that OpenAIChatCompletionsModel was created with the model and client
            MockModel.assert_called_once_with(
                model="claude-opus-4-1-20250805",
                openai_client=mock_client
            )
            
            # Check that Agent was created with the model instance
            MockAgent.assert_called_once_with(
                name="TestAgent",
                instructions="Test instructions",
                tools=[],
                model=mock_model,
                model_settings=None
            )
            assert agent == mock_agent
    
    def test_create_agent_ollama(self):
        """Test creating an Ollama agent."""
        with patch('nano_agent.modules.provider_config.AsyncOpenAI') as MockOpenAI, \
             patch('nano_agent.modules.provider_config.OpenAIChatCompletionsModel') as MockModel, \
             patch('nano_agent.modules.provider_config.Agent') as MockAgent:
            
            mock_client = Mock()
            MockOpenAI.return_value = mock_client
            
            mock_model = Mock()
            MockModel.return_value = mock_model
            
            mock_agent = Mock()
            MockAgent.return_value = mock_agent
            
            agent = ProviderConfig.create_agent(
                name="TestAgent",
                instructions="Test instructions",
                tools=[],
                model="gpt-oss:20b",
                provider="ollama",
                model_settings=None
            )
            
            MockOpenAI.assert_called_once_with(
                base_url="http://localhost:11434/v1",
                api_key="ollama"
            )
            
            MockModel.assert_called_once_with(
                model="gpt-oss:20b",
                openai_client=mock_client
            )
            
            MockAgent.assert_called_once_with(
                name="TestAgent",
                instructions="Test instructions",
                tools=[],
                model=mock_model,
                model_settings=None
            )
            assert agent == mock_agent
    
    def test_create_agent_invalid_provider(self):
        """Test creating an agent with invalid provider raises error."""
        with pytest.raises(ValueError, match="Unsupported provider: invalid"):
            ProviderConfig.create_agent(
                name="TestAgent",
                instructions="Test instructions",
                tools=[],
                model="some-model",
                provider="invalid",
                model_settings=None
            )
    
    def test_validate_provider_setup_valid_openai(self):
        """Test validation for valid OpenAI setup."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            is_valid, error = ProviderConfig.validate_provider_setup(
                "openai",
                "gpt-5-mini",
                AVAILABLE_MODELS,
                PROVIDER_REQUIREMENTS
            )
            assert is_valid is True
            assert error is None
    
    def test_validate_provider_setup_missing_api_key(self):
        """Test validation with missing API key."""
        with patch.dict(os.environ, {}, clear=True):
            is_valid, error = ProviderConfig.validate_provider_setup(
                "openai",
                "gpt-5-mini",
                AVAILABLE_MODELS,
                PROVIDER_REQUIREMENTS
            )
            assert is_valid is False
            assert "Missing environment variable: OPENAI_API_KEY" in error
    
    def test_validate_provider_setup_invalid_model(self):
        """Test validation with invalid model for provider."""
        is_valid, error = ProviderConfig.validate_provider_setup(
            "openai",
            "invalid-model",
            AVAILABLE_MODELS,
            PROVIDER_REQUIREMENTS
        )
        assert is_valid is False
        assert "Model invalid-model not available for openai" in error
    
    def test_validate_provider_setup_ollama_not_running(self):
        """Test validation when Ollama is not running."""
        with patch('nano_agent.modules.provider_config.requests.get') as mock_get:
            mock_get.side_effect = Exception("Connection refused")
            
            is_valid, error = ProviderConfig.validate_provider_setup(
                "ollama",
                "gpt-oss:20b",
                AVAILABLE_MODELS,
                PROVIDER_REQUIREMENTS
            )
            assert is_valid is False
            assert "Error checking Ollama availability" in error
    
    def test_setup_provider_disables_tracing_non_openai(self):
        """Test that tracing is disabled for non-OpenAI providers without key."""
        with patch.dict(os.environ, {}, clear=True), \
             patch('nano_agent.modules.provider_config.set_tracing_disabled') as mock_disable:
            
            ProviderConfig.setup_provider("anthropic")
            mock_disable.assert_called_once_with(True)
    
    def test_setup_provider_keeps_tracing_with_openai_key(self):
        """Test that tracing is kept when OpenAI key exists for non-OpenAI providers."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}), \
             patch('nano_agent.modules.provider_config.set_tracing_disabled') as mock_disable:
            
            ProviderConfig.setup_provider("anthropic")
            mock_disable.assert_not_called()
    
    def test_create_agent_azure(self):
        """Test creating an Azure OpenAI agent."""
        with patch('nano_agent.modules.provider_config.AsyncOpenAI') as MockOpenAI, \
             patch('nano_agent.modules.provider_config.OpenAIChatCompletionsModel') as MockModel, \
             patch('nano_agent.modules.provider_config.Agent') as MockAgent, \
             patch.dict(os.environ, {
                 'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com',
                 'AZURE_OPENAI_API_KEY': 'test-key',
                 'AZURE_OPENAI_DEPLOYMENT': 'gpt-5-mini'
             }):
            
            mock_client = Mock()
            MockOpenAI.return_value = mock_client
            
            mock_model = Mock()
            MockModel.return_value = mock_model
            
            mock_agent = Mock()
            MockAgent.return_value = mock_agent
            
            agent = ProviderConfig.create_agent(
                name="TestAgent",
                instructions="Test instructions",
                tools=[],
                model="gpt-5-mini",
                provider="azure",
                model_settings=None
            )
            
            MockOpenAI.assert_called_once_with(
                base_url="https://test.openai.azure.com/openai/deployments/gpt-5-mini",
                api_key="test-key",
                default_headers={"api-key": "test-key"},
                default_query={"api-version": "2024-05-01-preview"},
            )
            
            MockModel.assert_called_once_with(
                model="gpt-5-mini",
                openai_client=mock_client
            )
            
            MockAgent.assert_called_once_with(
                name="TestAgent",
                instructions="Test instructions",
                tools=[],
                model=mock_model,
                model_settings=None
            )
            assert agent == mock_agent
    
    def test_create_agent_azure_missing_endpoint(self):
        """Test creating Azure agent with missing endpoint raises error."""
        with patch.dict(os.environ, {'AZURE_OPENAI_API_KEY': 'test-key'}):
            with pytest.raises(ValueError, match="Missing AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY"):
                ProviderConfig.create_agent(
                    name="TestAgent",
                    instructions="Test instructions", 
                    tools=[],
                    model="gpt-5-mini",
                    provider="azure",
                    model_settings=None
                )
    
    def test_create_agent_azure_missing_api_key(self):
        """Test creating Azure agent with missing API key raises error."""
        with patch.dict(os.environ, {'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com'}):
            with pytest.raises(ValueError, match="Missing AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY"):
                ProviderConfig.create_agent(
                    name="TestAgent",
                    instructions="Test instructions",
                    tools=[],
                    model="gpt-5-mini", 
                    provider="azure",
                    model_settings=None
                )
    
    def test_validate_provider_setup_valid_azure(self):
        """Test validation for valid Azure setup."""
        with patch.dict(os.environ, {
            'AZURE_OPENAI_API_KEY': 'test-key',
            'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com'
        }):
            is_valid, error = ProviderConfig.validate_provider_setup(
                "azure",
                "gpt-5-mini",
                AVAILABLE_MODELS,
                PROVIDER_REQUIREMENTS
            )
            assert is_valid is True
            assert error is None
    
    def test_validate_provider_setup_azure_missing_api_key(self):
        """Test Azure validation with missing API key."""
        with patch.dict(os.environ, {}, clear=True):
            is_valid, error = ProviderConfig.validate_provider_setup(
                "azure",
                "gpt-5-mini",
                AVAILABLE_MODELS,
                PROVIDER_REQUIREMENTS
            )
            assert is_valid is False
            assert "Missing environment variable: AZURE_OPENAI_API_KEY" in error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])