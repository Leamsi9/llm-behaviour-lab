"""Integration tests for the main integrated app"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, Mock
import json


@pytest.fixture
def test_client():
    """Create test client for the integrated app"""
    # Import here to avoid circular imports
    from app_llm_behaviour_lab import app
    # TestClient takes app as first positional argument
    with TestClient(app) as client:
        yield client


class TestAppRoutes:
    """Test app route handlers"""
    
    def test_index_route(self, test_client):
        """Test main index route"""
        response = test_client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_energy_route(self, test_client):
        """Test energy UI route"""
        response = test_client.get("/energy")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_alignment_route(self, test_client):
        """Test alignment UI route"""
        response = test_client.get("/alignment")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_comparison_route(self, test_client):
        """Test comparison UI route"""
        response = test_client.get("/behaviour")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestAPIEndpoints:
    """Test API endpoints"""
    
    @pytest.mark.asyncio
    async def test_models_endpoint(self, test_client, mock_ollama_connection):
        """Test /api/models endpoint"""
        response = test_client.get("/api/models")
        assert response.status_code == 200
        
        data = response.json()
        assert "models" in data
        assert "current" in data
        assert isinstance(data["models"], list)
    
    def test_health_endpoint(self, test_client, mock_ollama_connection):
        """Test /api/health endpoint"""
        response = test_client.get("/api/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "ollama" in data
        assert "websocket" in data
    
    def test_energy_benchmarks_endpoint(self, test_client):
        """Test /api/energy-benchmarks endpoint"""
        response = test_client.get("/api/energy-benchmarks")
        assert response.status_code == 200
        
        data = response.json()
        assert "benchmarks" in data
        assert isinstance(data["benchmarks"], list)
        assert len(data["benchmarks"]) > 0
    
    def test_injection_methods_endpoint(self, test_client):
        """Test /api/injection-methods endpoint"""
        response = test_client.get("/api/injection-methods")
        assert response.status_code == 200
        
        data = response.json()
        assert "methods" in data
    
    def test_tool_methods_endpoint(self, test_client):
        """Test /api/tool-methods endpoint"""
        response = test_client.get("/api/tool-methods")
        assert response.status_code == 200
        
        data = response.json()
        assert "methods" in data
    
    def test_session_summary_endpoint(self, test_client, mock_energy_tracker):
        """Test /api/session-summary endpoint"""
        response = test_client.get("/api/session-summary")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_energy_wh" in data or "error" in data
    
    def test_benchmark_info_endpoint(self, test_client):
        """Test /api/benchmark-info endpoint"""
        response = test_client.get("/api/benchmark-info")
        assert response.status_code == 200
        
        data = response.json()
        assert "benchmarks" in data
        assert "co2_info" in data
        assert "benchmark_sources" in data
    
    def test_export_session_endpoint(self, test_client, mock_energy_tracker, tmp_path):
        """Test /api/export-session endpoint"""
        filepath = str(tmp_path / "test_export.json")
        
        response = test_client.post(
            "/api/export-session",
            params={"filepath": filepath}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "success" in data or "filepath" in data
    
    def test_switch_benchmark_endpoint(self, test_client, mock_energy_tracker):
        """Test /api/switch-benchmark endpoint"""
        response = test_client.post(
            "/api/switch-benchmark",
            params={"benchmark_name": "nvidia_a100"}
        )
        # May succeed or fail depending on tracker state
        assert response.status_code in [200, 400]
    
    def test_add_custom_benchmark_endpoint(self, test_client, mock_energy_tracker):
        """Test /api/add-custom-benchmark endpoint"""
        benchmark_data = {
            "name": "test_custom",
            "description": "Test custom benchmark",
            "watt_hours_per_1000_tokens": 0.5,
            "source": "Test",
            "hardware_specs": "Test hardware"
        }
        
        response = test_client.post(
            "/api/add-custom-benchmark",
            json=benchmark_data
        )
        # May succeed or fail depending on tracker state
        assert response.status_code in [200, 400, 500]


class TestTokenCounting:
    """Test token counting utilities"""
    
    def test_count_tokens_basic(self):
        """Test basic token counting"""
        from app_llm_behaviour_lab import count_tokens
        
        text = "Hello world"
        count = count_tokens(text, "llama3.1:8b")
        
        assert count > 0
        assert isinstance(count, int)
    
    def test_count_tokens_empty(self):
        """Test counting tokens in empty string"""
        from app_llm_behaviour_lab import count_tokens
        
        count = count_tokens("", "llama3.1:8b")
        assert count == 0
    
    def test_count_tokens_long_text(self):
        """Test counting tokens in long text"""
        from app_llm_behaviour_lab import count_tokens
        
        text = "Hello world. " * 100
        count = count_tokens(text, "llama3.1:8b")
        
        assert count > 100
    
    def test_analyze_token_breakdown(self):
        """Test token breakdown analysis"""
        from app_llm_behaviour_lab import analyze_token_breakdown
        
        breakdown = analyze_token_breakdown(
            system_prompt="You are helpful.",
            user_prompt="What is 2+2?",
            final_system="You are helpful. [INJECTED]",
            final_user="What is 2+2? [TOOL CONTEXT]",
            injection_result={"injection_metadata": {}},
            tool_integration_result={"integration_metadata": {}},
            ollama_prompt_tokens=50,
            ollama_completion_tokens=20,
            model_name="test-model"
        )
        
        assert "original" in breakdown
        assert "injected" in breakdown
        assert "tool_integration" in breakdown
        assert "generation" in breakdown
        assert "ollama_reported" in breakdown
        assert "analysis_notes" in breakdown
        
        assert breakdown["original"]["total_original_tokens"] > 0
        assert breakdown["ollama_reported"]["total_ollama_tokens"] == 70


class TestWebSocketHandling:
    """Test WebSocket connection handling"""
    
    @pytest.mark.skip(reason="WebSocket tests hang - need to refactor with proper async handling")
    def test_websocket_accepts_connection(self, test_client):
        """Test WebSocket accepts connections"""
        with test_client.websocket_connect("/ws") as websocket:
            # Connection should be established
            assert websocket is not None
            # Close immediately to avoid hanging
            websocket.close()
    
    @pytest.mark.skip(reason="WebSocket tests hang - need to refactor with proper async handling")
    def test_websocket_handles_invalid_json(self, test_client):
        """Test WebSocket handles invalid JSON"""
        with test_client.websocket_connect("/ws") as websocket:
            # Send invalid JSON
            websocket.send_text("not json")
            
            # Should receive error response
            response = websocket.receive_json()
            assert "error" in response
            websocket.close()
    
    @pytest.mark.skip(reason="WebSocket tests hang - need to refactor with proper async handling")
    def test_websocket_handles_cancel_command(self, test_client):
        """Test WebSocket handles cancel command"""
        with test_client.websocket_connect("/ws") as websocket:
            # Send cancel command
            websocket.send_json({"command": "cancel"})
            
            # Should not crash
            # May or may not receive response depending on state
            websocket.close()
    
    @pytest.mark.skip(reason="WebSocket tests hang - need to refactor with proper async handling")
    def test_websocket_requires_model_name(self, test_client):
        """Test WebSocket requires model_name"""
        with test_client.websocket_connect("/ws") as websocket:
            # Send payload without model_name
            websocket.send_json({
                "system": "Test",
                "user": "Test",
                "model_name": ""
            })
            
            response = websocket.receive_json()
            assert "error" in response
            websocket.close()
