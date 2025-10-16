"""Tests for ollama_client module"""
import pytest
from unittest.mock import AsyncMock, patch, Mock
import httpx

from ollama_client import (
    check_ollama_connection,
    list_ollama_models,
    get_models_with_defaults,
    print_startup_info,
    OLLAMA_BASE_URL,
)


class TestOllamaConnection:
    """Test Ollama connection functions"""
    
    @pytest.mark.asyncio
    async def test_check_ollama_connection_success(self):
        """Test successful Ollama connection"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            
            mock_get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.get = mock_get
            
            result = await check_ollama_connection()
            assert result is True
            mock_get.assert_called_once_with(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
    
    @pytest.mark.asyncio
    async def test_check_ollama_connection_failure(self):
        """Test failed Ollama connection"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get.side_effect = Exception("Connection error")
            
            result = await check_ollama_connection()
            assert result is False
    
    @pytest.mark.asyncio
    async def test_list_ollama_models_success(self):
        """Test listing Ollama models successfully"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "models": [
                    {"name": "llama3.1:8b"},
                    {"name": "mistral:7b"},
                ]
            }
            
            mock_get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.get = mock_get
            
            result = await list_ollama_models()
            assert result == ["llama3.1:8b", "mistral:7b"]
    
    @pytest.mark.asyncio
    async def test_list_ollama_models_empty(self):
        """Test listing models when none available"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"models": []}
            
            mock_get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.get = mock_get
            
            result = await list_ollama_models()
            assert result == []
    
    @pytest.mark.asyncio
    async def test_list_ollama_models_error(self):
        """Test listing models with connection error"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get.side_effect = Exception("Error")
            
            result = await list_ollama_models()
            assert result == []
    
    @pytest.mark.asyncio
    async def test_get_models_with_defaults(self):
        """Test getting models with default selection"""
        with patch('ollama_client.list_ollama_models', new_callable=AsyncMock) as mock_list:
            mock_list.return_value = ["model1", "model2", "model3"]
            
            result = await get_models_with_defaults()
            
            assert "models" in result
            assert "current" in result
            assert result["models"] == ["model1", "model2", "model3"]
            assert "base" in result["current"]
            assert "instruct" in result["current"]
    
    @pytest.mark.asyncio
    async def test_get_models_with_defaults_empty(self):
        """Test getting models when none available"""
        with patch('ollama_client.list_ollama_models', new_callable=AsyncMock) as mock_list, \
             patch('ollama_client.DEFAULT_MODEL', ""), \
             patch('ollama_client.DEFAULT_BASE_MODEL', ""):
            mock_list.return_value = []
            
            result = await get_models_with_defaults()
            
            assert result["models"] == []
            # When no models and no defaults, should return empty strings
            assert result["current"]["base"] == ""
            assert result["current"]["instruct"] == ""
    
    @pytest.mark.asyncio
    async def test_print_startup_info_connected(self, capsys):
        """Test startup info when Ollama is connected"""
        with patch('ollama_client.check_ollama_connection', new_callable=AsyncMock) as mock_check, \
             patch('ollama_client.list_ollama_models', new_callable=AsyncMock) as mock_list:
            mock_check.return_value = True
            mock_list.return_value = ["model1", "model2"]
            
            await print_startup_info("Test App")
            
            captured = capsys.readouterr()
            assert "Starting Test App" in captured.out
            assert "✓ Connected to Ollama" in captured.out
            assert "✓ Available models" in captured.out
    
    @pytest.mark.asyncio
    async def test_print_startup_info_disconnected(self, capsys):
        """Test startup info when Ollama is not connected"""
        with patch('ollama_client.check_ollama_connection', new_callable=AsyncMock) as mock_check:
            mock_check.return_value = False
            
            await print_startup_info("Test App")
            
            captured = capsys.readouterr()
            assert "Starting Test App" in captured.out
            assert "✗ Could not connect to Ollama" in captured.out
