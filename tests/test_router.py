"""Tests for the test router"""
import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from app_router import (
    AppRouter,
    test_router,
    determine_test_type,
    route_websocket_message
)


class TestAppRouter:
    """Test AppRouter class"""
    
    def test_router_initialization(self):
        """Test router initializes correctly"""
        router = AppRouter()
        assert router.cancel_events == {}
    
    def test_create_cancel_event(self):
        """Test creating cancel event"""
        router = AppRouter()
        event = router.create_cancel_event("session_1")
        
        assert isinstance(event, asyncio.Event)
        assert "session_1" in router.cancel_events
        assert router.cancel_events["session_1"] == event
    
    def test_get_cancel_event(self):
        """Test getting cancel event"""
        router = AppRouter()
        event = router.create_cancel_event("session_1")
        
        retrieved = router.get_cancel_event("session_1")
        assert retrieved == event
    
    def test_get_nonexistent_cancel_event(self):
        """Test getting nonexistent cancel event returns None"""
        router = AppRouter()
        assert router.get_cancel_event("nonexistent") is None
    
    def test_trigger_cancel(self):
        """Test triggering cancellation"""
        router = AppRouter()
        event = router.create_cancel_event("session_1")
        
        assert not event.is_set()
        router.trigger_cancel("session_1")
        assert event.is_set()
    
    def test_trigger_cancel_nonexistent(self):
        """Test triggering cancel on nonexistent session doesn't crash"""
        router = AppRouter()
        router.trigger_cancel("nonexistent")  # Should not raise
    
    def test_cleanup_session(self):
        """Test cleaning up session"""
        router = AppRouter()
        router.create_cancel_event("session_1")
        
        assert "session_1" in router.cancel_events
        router.cleanup_session("session_1")
        assert "session_1" not in router.cancel_events
    
    def test_cleanup_nonexistent_session(self):
        """Test cleaning up nonexistent session doesn't crash"""
        router = AppRouter()
        router.cleanup_session("nonexistent")  # Should not raise
    
    @pytest.mark.asyncio
    async def test_route_energy_test(self):
        """Test routing energy test"""
        router = AppRouter()
        mock_websocket = AsyncMock()
        cancel_event = asyncio.Event()
        
        with patch('app_router.run_energy_test', new_callable=AsyncMock) as mock_run:
            from app_energy import EnergyPayload
            payload = EnergyPayload(
                system="Test",
                user="Test",
                model_name="test-model",
                energy_benchmark="conservative_estimate"
            )
            
            await router.route_energy_test(payload, mock_websocket, cancel_event)
            
            mock_run.assert_called_once_with(payload, mock_websocket, cancel_event)
    
    @pytest.mark.asyncio
    async def test_route_alignment_test(self):
        """Test routing alignment test"""
        router = AppRouter()
        mock_websocket = AsyncMock()
        cancel_event = asyncio.Event()
        
        with patch('app_router.run_alignment_test', new_callable=AsyncMock) as mock_run:
            from app_alignment import AlignmentPayload
            payload = AlignmentPayload(
                system="Test",
                user="Test",
                model_name="test-model"
            )
            
            await router.route_alignment_test(payload, mock_websocket, cancel_event)
            
            mock_run.assert_called_once_with(payload, mock_websocket, cancel_event)
    
    @pytest.mark.asyncio
    async def test_route_comparison_test(self):
        """Test routing comparison test"""
        router = AppRouter()
        mock_websocket = AsyncMock()
        cancel_event = asyncio.Event()
        
        with patch('app_router.run_comparison_generation', new_callable=AsyncMock) as mock_run:
            from app_model_comparison import Payload as ComparisonPayload
            payload = ComparisonPayload(
                system="Test",
                user="Test",
                template="",
                model_name="test-model"
            )
            
            await router.route_comparison_test(payload, mock_websocket, cancel_event)
            
            mock_run.assert_called_once_with(payload, mock_websocket, cancel_event)


class TestDetermineTestType:
    """Test test type determination"""
    
    def test_determine_energy_type_with_benchmark(self):
        """Test detecting energy test type with energy_benchmark"""
        payload = {
            "system": "Test",
            "user": "Test",
            "model_name": "test-model",
            "energy_benchmark": "conservative_estimate"
        }
        assert determine_test_type(payload) == "energy"
    
    def test_determine_energy_type_with_injection(self):
        """Test detecting energy test type with injection and energy_benchmark"""
        payload = {
            "system": "Test",
            "user": "Test",
            "model_name": "test-model",
            "energy_benchmark": "conservative_estimate",
            "injection_type": "jailbreak"
        }
        assert determine_test_type(payload) == "energy"
    
    def test_determine_alignment_type_with_injection(self):
        """Test detecting alignment test type with injection_type"""
        payload = {
            "system": "Test",
            "user": "Test",
            "model_name": "test-model",
            "injection_type": "jailbreak"
        }
        assert determine_test_type(payload) == "alignment"
    
    def test_determine_alignment_type_with_tool(self):
        """Test detecting alignment test type with tool_integration_method"""
        payload = {
            "system": "Test",
            "user": "Test",
            "model_name": "test-model",
            "tool_integration_method": "inline"
        }
        assert determine_test_type(payload) == "alignment"
    
    def test_determine_comparison_type_with_use_base_model(self):
        """Test detecting comparison test type with use_base_model"""
        payload = {
            "system": "Test",
            "user": "Test",
            "template": "",
            "use_base_model": False
        }
        assert determine_test_type(payload) == "comparison"
    
    def test_determine_comparison_type_with_template(self):
        """Test detecting comparison test type with template"""
        payload = {
            "system": "Test",
            "user": "Test",
            "template": "some template"
        }
        assert determine_test_type(payload) == "comparison"
    
    def test_determine_default_alignment_with_model_name(self):
        """Test default to alignment when model_name present"""
        payload = {
            "system": "Test",
            "user": "Test",
            "model_name": "test-model"
        }
        assert determine_test_type(payload) == "alignment"
    
    def test_determine_default_comparison_fallback(self):
        """Test default fallback to comparison"""
        payload = {
            "system": "Test",
            "user": "Test"
        }
        assert determine_test_type(payload) == "comparison"


class TestRouteWebSocketMessage:
    """Test WebSocket message routing"""
    
    @pytest.mark.asyncio
    async def test_route_energy_message(self):
        """Test routing energy message"""
        mock_websocket = AsyncMock()
        cancel_event = asyncio.Event()
        
        payload_dict = {
            "system": "Test",
            "user": "Test",
            "model_name": "test-model",
            "energy_benchmark": "conservative_estimate"
        }
        
        with patch('app_router.test_router.route_energy_test', new_callable=AsyncMock) as mock_route:
            await route_websocket_message(payload_dict, mock_websocket, cancel_event)
            
            assert mock_route.called
            # Verify arguments were passed correctly (positional args)
            call_args = mock_route.call_args
            assert call_args[0][1] == mock_websocket  # Second positional arg
            assert call_args[0][2] == cancel_event  # Third positional arg
    
    @pytest.mark.asyncio
    async def test_route_alignment_message(self):
        """Test routing alignment message"""
        mock_websocket = AsyncMock()
        cancel_event = asyncio.Event()
        
        payload_dict = {
            "system": "Test",
            "user": "Test",
            "model_name": "test-model",
            "injection_type": "jailbreak"
        }
        
        with patch('app_router.test_router.route_alignment_test', new_callable=AsyncMock) as mock_route:
            await route_websocket_message(payload_dict, mock_websocket, cancel_event)
            
            assert mock_route.called
    
    @pytest.mark.asyncio
    async def test_route_comparison_message(self):
        """Test routing comparison message"""
        mock_websocket = AsyncMock()
        cancel_event = asyncio.Event()
        
        payload_dict = {
            "system": "Test",
            "user": "Test",
            "template": "",
            "use_base_model": False
        }
        
        with patch('app_router.test_router.route_comparison_test', new_callable=AsyncMock) as mock_route:
            await route_websocket_message(payload_dict, mock_websocket, cancel_event)
            
            assert mock_route.called


class TestGlobalRouterInstance:
    """Test global router instance"""
    
    def test_global_router_exists(self):
        """Test global router instance exists"""
        assert test_router is not None
        assert isinstance(test_router, AppRouter)
    
    def test_global_router_is_singleton(self):
        """Test global router is singleton"""
        from app_router import test_router as router2
        assert test_router is router2
    
    def test_global_router_is_app_router(self):
        """Test global router is AppRouter instance"""
        assert isinstance(test_router, AppRouter)
