"""Tests for standalone energy app"""
import pytest
from starlette.testclient import TestClient


@pytest.fixture
def energy_client():
    """Create test client for energy app"""
    from app_energy import app
    with TestClient(app) as client:
        yield client


class TestEnergyAppRoutes:
    """Test energy app route handlers"""
    
    def test_root_route(self, energy_client):
        """Test root route serves energy UI"""
        response = energy_client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_energy_route(self, energy_client):
        """Test energy UI route"""
        response = energy_client.get("/energy")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestEnergyAPIEndpoints:
    """Test energy API endpoints"""
    
    @pytest.mark.asyncio
    async def test_models_endpoint(self, energy_client):
        """Test /api/models endpoint"""
        response = energy_client.get("/api/models")
        assert response.status_code == 200
        
        data = response.json()
        assert "models" in data
        assert "current" in data or "defaults" in data
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self, energy_client):
        """Test /api/health endpoint"""
        response = energy_client.get("/api/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "ollama" in data
        assert "websocket" in data
        assert "models" in data
    
    @pytest.mark.asyncio
    async def test_energy_benchmarks_endpoint(self, energy_client):
        """Test /api/energy-benchmarks endpoint"""
        response = energy_client.get("/api/energy-benchmarks")
        assert response.status_code == 200
        
        data = response.json()
        assert "benchmarks" in data
        assert len(data["benchmarks"]) > 0
    
    @pytest.mark.asyncio
    async def test_injection_methods_endpoint(self, energy_client):
        """Test /api/injection-methods endpoint"""
        response = energy_client.get("/api/injection-methods")
        assert response.status_code == 200
        
        data = response.json()
        assert "methods" in data
    
    @pytest.mark.asyncio
    async def test_tool_methods_endpoint(self, energy_client):
        """Test /api/tool-methods endpoint"""
        response = energy_client.get("/api/tool-methods")
        assert response.status_code == 200
        
        data = response.json()
        assert "methods" in data
    
    @pytest.mark.asyncio
    async def test_session_summary_endpoint(self, energy_client):
        """Test /api/session-summary endpoint"""
        response = energy_client.get("/api/session-summary")
        assert response.status_code == 200
        
        # Should return error when no readings
        data = response.json()
        assert "error" in data or "total_energy_wh" in data


class TestEnergyPayload:
    """Test EnergyPayload dataclass"""
    
    def test_payload_creation(self):
        """Test creating an energy payload"""
        from app_energy import EnergyPayload
        
        payload = EnergyPayload(
            system="You are helpful",
            user="What is 2+2?",
            model_name="test-model",
            energy_benchmark="conservative_estimate"
        )
        
        assert payload.system == "You are helpful"
        assert payload.user == "What is 2+2?"
        assert payload.model_name == "test-model"
        assert payload.energy_benchmark == "conservative_estimate"
        assert payload.temp == 0.7  # default
        assert payload.max_tokens == 100  # default
    
    def test_payload_with_injection(self):
        """Test creating payload with injection config"""
        from app_energy import EnergyPayload
        
        payload = EnergyPayload(
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
        from app_energy import EnergyPayload
        
        payload = EnergyPayload(
            system="Test",
            user="Test",
            model_name="test-model",
            tool_integration_method="inline",
            tool_config={"tools": ["calculator"]}
        )
        
        assert payload.tool_integration_method == "inline"
        assert payload.tool_config == {"tools": ["calculator"]}
