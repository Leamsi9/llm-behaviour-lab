#!/usr/bin/env python3
"""
Router for LLM Behaviour Lab
Delegates requests to appropriate standalone apps based on test type
"""

import asyncio
from typing import Dict, Any, Optional
from fastapi import WebSocket

# Import standalone app test functions
from app_energy import EnergyPayload, run_energy_test
from app_alignment import AlignmentPayload, run_alignment_test
from app_model_comparison import Payload as ComparisonPayload, run_generation as run_comparison_generation


class AppRouter:
    """Routes test requests to appropriate standalone app logic"""
    
    def __init__(self):
        self.cancel_events: Dict[str, asyncio.Event] = {}
    
    async def route_energy_test(
        self,
        payload: EnergyPayload,
        websocket: WebSocket,
        cancel_event: asyncio.Event
    ):
        """
        Route energy test to standalone energy app logic
        
        Args:
            payload: Energy test payload
            websocket: WebSocket connection
            cancel_event: Event to signal cancellation
        """
        await run_energy_test(payload, websocket, cancel_event)
    
    async def route_alignment_test(
        self,
        payload: AlignmentPayload,
        websocket: WebSocket,
        cancel_event: asyncio.Event
    ):
        """
        Route alignment test to standalone alignment app logic
        
        Args:
            payload: Alignment test payload
            websocket: WebSocket connection
            cancel_event: Event to signal cancellation
        """
        await run_alignment_test(payload, websocket, cancel_event)
    
    async def route_comparison_test(
        self,
        payload: ComparisonPayload,
        websocket: WebSocket,
        cancel_event: asyncio.Event
    ):
        """
        Route comparison test to standalone comparison app logic
        
        Args:
            payload: Comparison test payload
            websocket: WebSocket connection
            cancel_event: Event to signal cancellation
        """
        await run_comparison_generation(payload, websocket, cancel_event)
    
    def create_cancel_event(self, session_id: str) -> asyncio.Event:
        """Create and store a cancel event for a session"""
        event = asyncio.Event()
        self.cancel_events[session_id] = event
        return event
    
    def get_cancel_event(self, session_id: str) -> Optional[asyncio.Event]:
        """Get cancel event for a session"""
        return self.cancel_events.get(session_id)
    
    def trigger_cancel(self, session_id: str):
        """Trigger cancellation for a session"""
        if session_id in self.cancel_events:
            self.cancel_events[session_id].set()
    
    def cleanup_session(self, session_id: str):
        """Clean up session resources"""
        if session_id in self.cancel_events:
            del self.cancel_events[session_id]


# Global router instance
test_router = AppRouter()


def determine_test_type(payload: Dict[str, Any]) -> str:
    """
    Determine test type from payload
    
    Args:
        payload: Raw payload dictionary
        
    Returns:
        Test type: 'energy', 'alignment', or 'comparison'
    """
    # Check for energy-specific fields
    if 'energy_benchmark' in payload:
        return 'energy'
    
    # Check for alignment-specific fields (both have injection/tool but energy has energy_benchmark)
    if ('injection_type' in payload or 'tool_integration_method' in payload) and 'energy_benchmark' not in payload:
        return 'alignment'
    
    # Check for comparison-specific fields
    if 'use_base_model' in payload or 'template' in payload:
        return 'comparison'
    
    # Default to alignment if has model_name
    if 'model_name' in payload:
        return 'alignment'
    
    # Default fallback
    return 'comparison'


async def route_websocket_message(
    payload_dict: Dict[str, Any],
    websocket: WebSocket,
    cancel_event: asyncio.Event
):
    """
    Route WebSocket message to appropriate handler
    
    Args:
        payload_dict: Raw payload dictionary
        websocket: WebSocket connection
        cancel_event: Cancellation event
    """
    test_type = determine_test_type(payload_dict)
    print(f"🔀 [ROUTER] Routing request to: {test_type.upper()} app")
    await websocket.send_json({"log": f"🔀 [ROUTER] Routing request to: {test_type.upper()} app"})
    
    if test_type == 'energy':
        payload = EnergyPayload(**payload_dict)
        await test_router.route_energy_test(payload, websocket, cancel_event)
    
    elif test_type == 'alignment':
        payload = AlignmentPayload(**payload_dict)
        await test_router.route_alignment_test(payload, websocket, cancel_event)
    
    elif test_type == 'comparison':
        payload = ComparisonPayload(**payload_dict)
        await test_router.route_comparison_test(payload, websocket, cancel_event)
    
    else:
        await websocket.send_json({
            "error": f"Unknown test type: {test_type}"
        })
