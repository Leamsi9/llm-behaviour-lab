"""Pytest configuration and shared fixtures"""
import pytest
import asyncio
import httpx
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List


@pytest.fixture
def mock_ollama_response():
    """Mock Ollama streaming response"""
    return {
        "model": "test-model",
        "message": {"role": "assistant", "content": "Test response"},
        "done": True,
        "prompt_eval_count": 50,
        "eval_count": 20,
    }


@pytest.fixture
def mock_ollama_stream():
    """Mock Ollama streaming chunks"""
    return [
        '{"message": {"content": "Hello"}, "done": false}\n',
        '{"message": {"content": " world"}, "done": false}\n',
        '{"message": {"content": "!"}, "done": true, "prompt_eval_count": 50, "eval_count": 20}\n',
    ]


@pytest.fixture
def sample_payload():
    """Sample test payload"""
    return {
        "system": "You are a helpful assistant.",
        "user": "What is 2+2?",
        "model_name": "test-model",
        "strategy_name": "baseline",
        "energy_benchmark": "conservative_estimate",
        "injection_type": "none",
        "injection_params": {},
        "tool_integration_method": "none",
        "tool_config": {},
        "temp": 0.7,
        "max_tokens": 100,
    }


@pytest.fixture
def sample_energy_payload():
    """Sample energy test payload"""
    return {
        "system": "You are a helpful assistant.",
        "user": "What is 2+2?",
        "model_name": "test-model",
        "strategy_name": "baseline",
        "energy_benchmark": "conservative_estimate",
        "injection_type": "none",
        "injection_params": {},
        "tool_integration_method": "none",
        "tool_config": {},
        "temp": 0.7,
        "max_tokens": 100,
    }


@pytest.fixture
def sample_alignment_payload():
    """Sample alignment test payload"""
    return {
        "system": "You are a helpful assistant.",
        "user": "What is 2+2?",
        "model_name": "test-model",
        "strategy_name": "baseline",
        "injection_type": "none",
        "injection_params": {},
        "tool_integration_method": "none",
        "tool_config": {},
        "temp": 0.7,
        "max_tokens": 100,
    }


@pytest.fixture
def mock_websocket():
    """Mock WebSocket connection"""
    ws = AsyncMock()
    ws.accept = AsyncMock()
    ws.send_json = AsyncMock()
    ws.receive_json = AsyncMock()
    ws.close = AsyncMock()
    return ws


@pytest.fixture
def mock_cancel_event():
    """Mock asyncio Event for cancellation"""
    event = asyncio.Event()
    return event


@pytest.fixture
def mock_httpx_client():
    """Mock httpx AsyncClient"""
    client = AsyncMock()
    return client


@pytest.fixture
def mock_ollama_connection(monkeypatch):
    """Mock Ollama connection check"""
    async def mock_check():
        return True
    
    async def mock_list_models():
        return ["test-model-1", "test-model-2", "llama3.1:8b"]
    
    monkeypatch.setattr("ollama_client.check_ollama_connection", mock_check)
    monkeypatch.setattr("ollama_client.list_ollama_models", mock_list_models)
    
    return {"check": mock_check, "list": mock_list_models}


@pytest.fixture
def mock_energy_tracker(monkeypatch):
    """Mock energy tracker"""
    mock_tracker = Mock()
    mock_tracker.set_benchmark = Mock()
    mock_tracker.record_usage = Mock(return_value=Mock(
        benchmark_used="conservative_estimate",
        watt_hours_consumed=0.0123,
        carbon_grams_co2=0.00492,
        total_tokens=70,
    ))
    mock_tracker.get_session_summary = Mock(return_value={
        "total_energy_wh": 0.0123,
        "total_carbon_gco2": 0.00492,
        "total_tokens": 70,
        "average_energy_per_1000_tokens": 0.1757,
        "benchmark_used": "conservative_estimate",
    })
    
    monkeypatch.setattr("energy_tracker.energy_tracker", mock_tracker)
    return mock_tracker


@pytest.fixture
def mock_alignment_analyzer(monkeypatch):
    """Mock alignment analyzer"""
    from datetime import datetime
    
    mock_analyzer = Mock()
    mock_score = Mock(
        timestamp=datetime.now(),
        goal_adherence=0.95,
        consistency=0.90,
        relevance=0.92,
        factual_accuracy=0.88,
        hallucination_score=0.05,
        coherence_score=0.93,
        completeness_score=0.87,
        injection_bleed=0.0,
        tool_interference=0.0,
        off_topic_penalty=0.0,
        analysis_notes=["Response is well-aligned"],
        detected_issues=[],
    )
    mock_analyzer.analyze_response = Mock(return_value=mock_score)
    
    monkeypatch.setattr("alignment_analyzer.alignment_analyzer", mock_analyzer)
    return mock_analyzer


@pytest.fixture
def mock_injection_manager(monkeypatch):
    """Mock injection manager"""
    mock_manager = Mock()
    mock_manager.apply_injection = Mock(return_value={
        "modified_system": "You are a helpful assistant.",
        "modified_user": "What is 2+2?",
        "injection_metadata": {
            "injection_type": "none",
            "tokens_added": 0,
        }
    })
    mock_manager.get_available_injections = Mock(return_value=[
        {"name": "none", "description": "No injection"},
        {"name": "jailbreak", "description": "Jailbreak attempt"},
    ])
    
    monkeypatch.setattr("prompt_injection.injection_manager", mock_manager)
    return mock_manager


@pytest.fixture
def mock_tool_manager(monkeypatch):
    """Mock tool manager"""
    mock_manager = Mock()
    mock_manager.integrate_tools = AsyncMock(return_value={
        "modified_prompts": {
            "system": "You are a helpful assistant.",
            "user": "What is 2+2?",
        },
        "integration_metadata": {
            "method": "none",
            "tools_processed": 0,
        }
    })
    mock_manager.get_available_methods = Mock(return_value=[
        {"name": "none", "description": "No tool integration"},
        {"name": "inline", "description": "Inline tool results"},
    ])
    
    monkeypatch.setattr("tool_integration.tool_manager", mock_manager)
    return mock_manager


@pytest.fixture(scope="session")
async def real_ollama_models() -> List[str]:
    """Get real models from Ollama if available, otherwise return test models"""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = [model["name"] for model in data.get("models", [])]
                if models:
                    return models
    except Exception:
        pass
    
    # Fallback to common test model names
    return ["llama3.1:8b", "mistral:7b", "gemma2:9b"]


@pytest.fixture
def use_real_model(real_ollama_models):
    """Get a real model name for testing"""
    # Return the first available model
    return real_ollama_models[0] if real_ollama_models else "test-model"
