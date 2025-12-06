"""Tests for the model comparison app"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, Mock
import json


@pytest.fixture
def comparison_client():
    """Create test client for the comparison app"""
    from app_model_comparison import app
    # TestClient takes app as first positional argument
    with TestClient(app) as client:
        yield client


class TestComparisonAppRoutes:
    """Test comparison app route handlers"""
    
    def test_index_route(self, comparison_client):
        """Test main index route redirects to comparison UI"""
        response = comparison_client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_comparison_route(self, comparison_client):
        """Test comparison UI route"""
        # The comparison app serves UI at root, not /behaviour
        response = comparison_client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestComparisonAPIEndpoints:
    """Test comparison API endpoints"""
    
    @pytest.mark.asyncio
    async def test_models_endpoint(self, comparison_client):
        """Test /api/models endpoint"""
        with patch('app_model_comparison.list_ollama_models', new_callable=AsyncMock) as mock_list:
            mock_list.return_value = ["model1", "model2", "model3"]
            
            response = comparison_client.get("/api/models")
            assert response.status_code == 200
            
            data = response.json()
            assert "models" in data
            assert "current" in data
            assert isinstance(data["models"], list)
    
    def test_health_endpoint(self, comparison_client):
        """Test /api/health endpoint"""
        with patch('app_model_comparison.check_ollama_connection', new_callable=AsyncMock) as mock_check:
            mock_check.return_value = True
            
            response = comparison_client.get("/api/health")
            assert response.status_code == 200
            
            data = response.json()
            assert "status" in data
            assert "ollama" in data


class TestComparisonWebSocket:
    """Test comparison WebSocket functionality"""
    
    @pytest.mark.skip(reason="WebSocket tests hang - need to refactor with proper async handling")
    @pytest.mark.asyncio
    async def test_websocket_accepts_connection(self, comparison_client):
        """Test WebSocket accepts connections"""
        with comparison_client.websocket_connect("/ws") as websocket:
            assert websocket is not None
    
    @pytest.mark.skip(reason="WebSocket tests hang - need to refactor with proper async handling")
    @pytest.mark.asyncio
    async def test_websocket_handles_invalid_json(self, comparison_client):
        """Test WebSocket handles invalid JSON"""
        with comparison_client.websocket_connect("/ws") as websocket:
            websocket.send_text("not json")
            response = websocket.receive_json()
            assert "error" in response
    
    @pytest.mark.skip(reason="WebSocket tests hang - need to refactor with proper async handling")
    @pytest.mark.asyncio
    async def test_websocket_requires_model_name(self, comparison_client):
        """Test WebSocket requires model_name"""
        with comparison_client.websocket_connect("/ws") as websocket:
            websocket.send_json({
                "system": "Test",
                "user": "Test",
                "model_name": "",
                "model_key": "base"
            })
            
            response = websocket.receive_json()
            assert "error" in response
    
    @pytest.mark.skip(reason="WebSocket tests hang - need to refactor with proper async handling")
    @pytest.mark.asyncio
    async def test_websocket_cancel_command(self, comparison_client):
        """Test WebSocket cancel command"""
        with comparison_client.websocket_connect("/ws") as websocket:
            websocket.send_json({
                "command": "cancel",
                "model_key": "base"
            })
            # Should not crash


class TestComparisonPayload:
    """Test Payload dataclass"""
    
    def test_payload_creation(self):
        """Test creating a comparison payload"""
        from app_model_comparison import Payload
        
        payload = Payload(
            system="You are helpful",
            user="What is 2+2?",
            template="",
            model_name="test-model",
            use_base_model=False
        )
        
        assert payload.system == "You are helpful"
        assert payload.user == "What is 2+2?"
        assert payload.model_name == "test-model"
        assert payload.use_base_model == False
        assert payload.temp == 0.7  # default
        assert payload.max_tokens == 1024  # default


class TestComparisonGeneration:
    """Test generation logic"""
    
    @pytest.mark.skip(reason="Generation tests need complex mocking - to be implemented")
    @pytest.mark.asyncio
    async def test_run_generation_basic(self, mock_websocket, mock_cancel_event):
        """Test basic generation run"""
        from app_model_comparison import run_generation, Payload
        
        payload = Payload(
            system="You are helpful",
            user="What is 2+2?",
            model_name="test-model",
            model_key="base"
        )
        
        # Mock httpx client
        with patch('app_model_comparison.httpx.AsyncClient') as mock_client:
            # Mock streaming response
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.aiter_lines = AsyncMock(return_value=iter([
                '{"message": {"content": "4"}, "done": false}',
                '{"message": {"content": ""}, "done": true, "prompt_eval_count": 10, "eval_count": 5}'
            ]))
            mock_response.aclose = AsyncMock()
            
            mock_stream = AsyncMock()
            mock_stream.__aenter__.return_value = mock_response
            mock_stream.__aexit__.return_value = None
            
            mock_client_instance = AsyncMock()
            mock_client_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            mock_client.return_value.__aexit__.return_value = None
            
            await run_generation(payload, mock_websocket, mock_cancel_event)
            
            # Should have sent tokens
            assert mock_websocket.send_json.called
            
            # Check that final message was sent
            calls = mock_websocket.send_json.call_args_list
            final_call = calls[-1][0][0]
            assert final_call["done"] is True
            assert final_call["model_key"] == "base"
    
    @pytest.mark.skip(reason="Generation tests need complex mocking - to be implemented")
    @pytest.mark.asyncio
    async def test_run_generation_with_cancellation(self, mock_websocket):
        """Test generation with cancellation"""
        from app_model_comparison import run_generation, Payload
        import asyncio
        
        payload = Payload(
            system="You are helpful",
            user="What is 2+2?",
            model_name="test-model",
            model_key="base"
        )
        
        # Create cancel event and set it immediately
        cancel_event = asyncio.Event()
        cancel_event.set()
        
        with patch('app_model_comparison.httpx.AsyncClient') as mock_client:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.aiter_lines = AsyncMock(return_value=iter([]))
            mock_response.aclose = AsyncMock()
            
            mock_stream = AsyncMock()
            mock_stream.__aenter__.return_value = mock_response
            mock_stream.__aexit__.return_value = None
            
            mock_client_instance = AsyncMock()
            mock_client_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            mock_client.return_value.__aexit__.return_value = None
            
            await run_generation(payload, mock_websocket, cancel_event)
            
            # Should have sent cancellation message
            calls = mock_websocket.send_json.call_args_list
            assert any("cancelled" in str(call) for call in calls)
    
    @pytest.mark.skip(reason="Generation tests need complex mocking - to be implemented")
    @pytest.mark.asyncio
    async def test_run_generation_connection_error(self, mock_websocket, mock_cancel_event):
        """Test generation with connection error"""
        from app_model_comparison import run_generation, Payload
        import httpx
        
        payload = Payload(
            system="You are helpful",
            user="What is 2+2?",
            model_name="test-model",
            model_key="base"
        )
        
        with patch('app_model_comparison.httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.side_effect = httpx.ConnectError("Connection failed")
            
            await run_generation(payload, mock_websocket, mock_cancel_event)
            
            # Should have sent error message
            calls = mock_websocket.send_json.call_args_list
            assert any("error" in call[0][0] for call in calls)
    
    @pytest.mark.skip(reason="Generation tests need complex mocking - to be implemented")
    @pytest.mark.asyncio
    async def test_run_generation_ollama_error(self, mock_websocket, mock_cancel_event):
        """Test generation with Ollama error response"""
        from app_model_comparison import run_generation, Payload
        
        payload = Payload(
            system="You are helpful",
            user="What is 2+2?",
            model_name="nonexistent-model",
            model_key="base"
        )
        
        with patch('app_model_comparison.httpx.AsyncClient') as mock_client:
            mock_response = AsyncMock()
            mock_response.status_code = 404
            mock_response.aread = AsyncMock(return_value=b"Model not found")
            
            mock_stream = AsyncMock()
            mock_stream.__aenter__.return_value = mock_response
            mock_stream.__aexit__.return_value = None
            
            mock_client_instance = AsyncMock()
            mock_client_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            mock_client.return_value.__aexit__.return_value = None
            
            await run_generation(payload, mock_websocket, mock_cancel_event)
            
            # Should have sent error message
            calls = mock_websocket.send_json.call_args_list
            error_call = calls[-1][0][0]
            assert "error" in error_call
            assert "Ollama error" in error_call["error"]


class TestComparisonUtilities:
    """Test utility functions"""
    
    @pytest.mark.asyncio
    async def test_check_ollama_connection(self):
        """Test checking Ollama connection"""
        from app_model_comparison import check_ollama_connection
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.get = mock_get
            
            result = await check_ollama_connection()
            assert result is True
    
    @pytest.mark.asyncio
    async def test_list_ollama_models(self):
        """Test listing Ollama models"""
        from app_model_comparison import list_ollama_models
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "models": [
                    {"name": "model1"},
                    {"name": "model2"}
                ]
            }
            mock_get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.get = mock_get
            
            result = await list_ollama_models()
            assert result == ["model1", "model2"]
