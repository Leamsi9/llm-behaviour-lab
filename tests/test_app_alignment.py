"""Tests for standalone alignment app"""
import pytest
from starlette.testclient import TestClient


@pytest.fixture
def alignment_client():
    """Create test client for alignment app"""
    from app_alignment import app
    with TestClient(app) as client:
        yield client


class TestAlignmentAppRoutes:
    """Test alignment app route handlers"""
    
    def test_root_route(self, alignment_client):
        """Test root route serves alignment UI"""
        response = alignment_client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_alignment_route(self, alignment_client):
        """Test alignment UI route"""
        response = alignment_client.get("/alignment")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestAlignmentAPIEndpoints:
    """Test alignment API endpoints"""
    
    @pytest.mark.asyncio
    async def test_models_endpoint(self, alignment_client):
        """Test /api/models endpoint"""
        response = alignment_client.get("/api/models")
        assert response.status_code == 200
        
        data = response.json()
        assert "models" in data
        assert "current" in data or "defaults" in data
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self, alignment_client):
        """Test /api/health endpoint"""
        response = alignment_client.get("/api/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "ollama" in data
        assert "websocket" in data
        assert "models" in data
    
    @pytest.mark.asyncio
    async def test_injection_methods_endpoint(self, alignment_client):
        """Test /api/injection-methods endpoint"""
        response = alignment_client.get("/api/injection-methods")
        assert response.status_code == 200
        
        data = response.json()
        assert "methods" in data
    
    @pytest.mark.asyncio
    async def test_tool_methods_endpoint(self, alignment_client):
        """Test /api/tool-methods endpoint"""
        response = alignment_client.get("/api/tool-methods")
        assert response.status_code == 200
        
        data = response.json()
        assert "methods" in data


class TestAlignmentPayload:
    """Test AlignmentPayload dataclass"""
    
    def test_payload_creation(self):
        """Test creating an alignment payload"""
        from app_alignment import AlignmentPayload
        
        payload = AlignmentPayload(
            system="You are helpful",
            user="What is 2+2?",
            model_name="test-model"
        )
        
        assert payload.system == "You are helpful"
        assert payload.user == "What is 2+2?"
        assert payload.model_name == "test-model"
        assert payload.temp == 0.7  # default
        assert payload.max_tokens == 100  # default
    
    def test_payload_with_injection(self):
        """Test creating payload with injection config"""
        from app_alignment import AlignmentPayload
        
        payload = AlignmentPayload(
            system="Test",
            user="Test",
            model_name="test-model",
            injection_type="jailbreak",
            injection_params={"intensity": "high"}
        )
        
        assert payload.injection_type == "jailbreak"
        assert payload.injection_params == {"intensity": "high"}
    
    def test_payload_with_tool_integration(self):
        """Test creating payload with tool integration"""
        from app_alignment import AlignmentPayload
        
        payload = AlignmentPayload(
            system="Test",
            user="Test",
            model_name="test-model",
            tool_integration_method="inline",
            tool_config={"tools": ["calculator"]}
        )
        
        assert payload.tool_integration_method == "inline"
        assert payload.tool_config == {"tools": ["calculator"]}
    
    def test_payload_with_strategy(self):
        """Test creating payload with strategy name"""
        from app_alignment import AlignmentPayload
        
        payload = AlignmentPayload(
            system="Test",
            user="Test",
            model_name="test-model",
            strategy_name="jailbreak_inline"
        )
        
        assert payload.strategy_name == "jailbreak_inline"
