"""
Test to validate the default provider configuration.

This test ensures that the default provider is set to Azure as requested.
"""

import pytest
from nano_agent.modules.constants import DEFAULT_PROVIDER, AVAILABLE_MODELS, PROVIDER_REQUIREMENTS


class TestDefaultProvider:
    """Test default provider configuration."""
    
    def test_default_provider_is_azure(self):
        """Test that the default provider is set to azure."""
        assert DEFAULT_PROVIDER == "azure", f"Expected DEFAULT_PROVIDER to be 'azure', got '{DEFAULT_PROVIDER}'"
    
    def test_azure_provider_is_supported(self):
        """Test that azure is a supported provider."""
        assert "azure" in AVAILABLE_MODELS, "Azure should be in AVAILABLE_MODELS"
        assert "azure" in PROVIDER_REQUIREMENTS, "Azure should be in PROVIDER_REQUIREMENTS"
    
    def test_azure_has_models(self):
        """Test that azure provider has available models."""
        azure_models = AVAILABLE_MODELS.get("azure", [])
        assert len(azure_models) > 0, "Azure should have at least one available model"
        
        # Test that default model is available for azure
        from nano_agent.modules.constants import DEFAULT_MODEL
        assert DEFAULT_MODEL in azure_models, f"Default model '{DEFAULT_MODEL}' should be available for Azure provider"
    
    def test_azure_api_key_requirement(self):
        """Test that azure has correct API key requirement."""
        azure_requirement = PROVIDER_REQUIREMENTS.get("azure")
        assert azure_requirement == "AZURE_OPENAI_API_KEY", f"Expected Azure to require 'AZURE_OPENAI_API_KEY', got '{azure_requirement}'"